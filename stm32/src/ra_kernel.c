#include "ra_kernel.h"

static int16_t ra_table[257];
static int ra_table_ready;

static inline int16_t ra_sat16(int64_t v)
{
    if (v > 32767) return (int16_t)32767;
    if (v < -32768) return (int16_t)-32768;
    return (int16_t)v;
}

static int32_t ra_round_away(double v)
{
    return (int32_t)(v >= 0.0 ? (v + 0.5) : (v - 0.5));
}

void ra_tables_init(void)
{
    if (ra_table_ready) return;
    for (int i = 0; i < 256; ++i) {
        /* exp(-x) evaluated as 1/exp(+x). The direct series for exp(-x) is
           alternating, and near x = -8 its terms reach ~400 while the sum is
           ~3e-4, so catastrophic cancellation made entries 247..255 come out
           NEGATIVE -- they rounded to -257, one LSB below the RTL ROM and
           outside ELU's mathematical range of >= -1.0. Summing the positive
           series and taking the reciprocal has no cancellation and reproduces
           all 256 ROM entries exactly. Verified against rl_elu_array8.sv by
           rl_accel/qat/scripts/verify_against_rtl.py. */
        double x = ((double)i) / 32.0;
        double e = 1.0, term = 1.0;
        for (int n = 1; n < 24; ++n) { term *= x / (double)n; e += term; }
        ra_table[i] = (int16_t)ra_round_away((1.0 / e - 1.0) * 256.0);
    }
    ra_table[256] = -256;
    ra_table_ready = 1;
}

static inline int16_t ra_elu(int16_t x)
{
    if (x >= 0) return x;
    if (x <= -(8 * 256)) return (int16_t)(-256);
    int32_t mag = -(int32_t)x;
    int32_t idx = mag >> 3;
    if (idx > 255) idx = 255;
    int32_t rem = mag & 7;
    int32_t base = ra_table[idx];
    int32_t next = ra_table[idx + 1];
    return ra_sat16(base + (((next - base) * rem + 4) >> 3));
}

static inline int64_t ra_round_shift(int64_t v, uint8_t shift)
{
    if (shift == 0) return v;
    int64_t half = (int64_t)1 << (shift - 1);
    return (v + ((v < 0) ? -half : half)) >> shift;
}

void ra_quantize_obs(const float *obs, int16_t *obs_q, uint16_t count)
{
    for (uint16_t i = 0; i < count; ++i) {
        float s = obs[i] * 256.0f;
        int32_t r = (int32_t)(s >= 0.0f ? (s + 0.5f) : (s - 0.5f));
        obs_q[i] = ra_sat16(r);
    }
}

void ra_dequantize_act(const int16_t *act_q, float *act, uint16_t count)
{
    for (uint16_t i = 0; i < count; ++i) act[i] = (float)act_q[i] * (1.0f / 256.0f);
}

void ra_infer_scalar(const ra_model_t *m, const int16_t *obs_q, int16_t *act_q,
                     int16_t *scratch)
{
    int16_t *in = scratch, *out = scratch + m->scratch_elements;
    for (uint16_t i = 0; i < m->obs_features; ++i) in[i] = obs_q[i];
    for (uint16_t l = 0; l < m->n_layers; ++l) {
        const ra_layer_t *L = &m->layers[l];
        for (uint16_t o = 0; o < L->out_features; ++o) {
            const int16_t *row = L->weights + (size_t)o * L->row_stride;
            int64_t acc = 0;
            for (uint16_t k = 0; k < L->in_features; ++k)
                acc += (int32_t)row[k] * (int32_t)in[k];
            int16_t y = ra_sat16(ra_round_shift(acc, L->weight_frac) + L->bias[o]);
            out[o] = L->apply_elu ? ra_elu(y) : y;
        }
        int16_t *t = in; in = out; out = t;
    }
    for (uint16_t i = 0; i < m->act_features; ++i) act_q[i] = in[i];
}

/* SMLALD: signed dual 16x16 multiply with 64-bit accumulate, one instruction
   per two elements. Rows and the activation buffer are padded to a multiple of
   four int16 and 8-byte aligned, so the tail needs no special case. */
static inline int64_t ra_smlald(uint32_t a, uint32_t b, int64_t acc)
{
    union { int64_t q; struct { uint32_t lo, hi; } w; } u;
    u.q = acc;
    __asm volatile("smlald %0, %1, %2, %3"
                   : "+r"(u.w.lo), "+r"(u.w.hi) : "r"(a), "r"(b));
    return u.q;
}

void ra_infer_simd(const ra_model_t *m, const int16_t *obs_q, int16_t *act_q,
                   int16_t *scratch)
{
    int16_t *in = scratch, *out = scratch + m->scratch_elements;
    for (uint16_t i = 0; i < m->obs_features; ++i) in[i] = obs_q[i];
    /* Zero the padding lanes so they contribute nothing. */
    for (uint16_t i = m->obs_features; i < m->scratch_elements; ++i) in[i] = 0;

    for (uint16_t l = 0; l < m->n_layers; ++l) {
        const ra_layer_t *L = &m->layers[l];
        const uint16_t pairs = (uint16_t)(L->row_stride >> 1);
        for (uint16_t o = 0; o < L->out_features; ++o) {
            const uint32_t *row = (const uint32_t *)(const void *)
                                  (L->weights + (size_t)o * L->row_stride);
            const uint32_t *act = (const uint32_t *)(const void *)in;
            int64_t acc = 0;
            uint16_t p = 0;
            for (; p + 4 <= pairs; p += 4) {
                acc = ra_smlald(row[p + 0], act[p + 0], acc);
                acc = ra_smlald(row[p + 1], act[p + 1], acc);
                acc = ra_smlald(row[p + 2], act[p + 2], acc);
                acc = ra_smlald(row[p + 3], act[p + 3], acc);
            }
            for (; p < pairs; ++p) acc = ra_smlald(row[p], act[p], acc);
            int16_t y = ra_sat16(ra_round_shift(acc, L->weight_frac) + L->bias[o]);
            out[o] = L->apply_elu ? ra_elu(y) : y;
        }
        for (uint16_t i = L->out_features; i < m->scratch_elements; ++i) out[i] = 0;
        int16_t *t = in; in = out; out = t;
    }
    for (uint16_t i = 0; i < m->act_features; ++i) act_q[i] = in[i];
}
