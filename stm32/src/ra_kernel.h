/* RoboAccel policy inference for Cortex-M7, bit-exact with the FPGA.
 *
 * Q8.8 activations and bias, per-GEMM INT16 weight fractional bits,
 * round-half-away-from-zero requantize, INT16 saturation, Q8.8 ELU table with
 * 3-bit interpolation -- the same arithmetic tools/export_policy.py and the PL
 * implement. Two kernels are provided and must agree exactly:
 *   ra_infer_scalar  plain C
 *   ra_infer_simd    SMLALD dual 16x16 MAC, the M7's best integer path
 */
#ifndef RA_KERNEL_H
#define RA_KERNEL_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    const int16_t *weights;    /* [out_features][row_stride], zero padded */
    const int16_t *bias;
    uint16_t in_features;
    uint16_t out_features;
    uint16_t row_stride;       /* multiple of 4 so every row is 8-byte aligned */
    uint8_t  weight_frac;
    uint8_t  apply_elu;
} ra_layer_t;

typedef struct {
    const ra_layer_t *layers;
    uint16_t n_layers;
    uint16_t obs_features;
    uint16_t act_features;
    uint16_t scratch_elements;
} ra_model_t;

void ra_tables_init(void);
void ra_infer_scalar(const ra_model_t *m, const int16_t *obs_q, int16_t *act_q, int16_t *scratch);
void ra_infer_simd(const ra_model_t *m, const int16_t *obs_q, int16_t *act_q, int16_t *scratch);
void ra_quantize_obs(const float *obs, int16_t *obs_q, uint16_t count);
void ra_dequantize_act(const int16_t *act_q, float *act, uint16_t count);

#endif
