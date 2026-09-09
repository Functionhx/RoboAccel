/* RoboAccel policy benchmark on STM32H723VGT6 (DM-MC02 board).
 *
 * Clock, supply and flash latency reproduce the vendor's own configuration
 * from dm-mc02 CtrBoard-H7: LDO supply, VOS0, HSE 24 MHz, PLL1 M=2 N=40 P=1
 * -> SYSCLK 480 MHz, HCLK 240 MHz, FLASH_LATENCY_3. I-cache and D-cache on.
 *
 * Timing uses DWT CYCCNT, which counts CPU cycles at 480 MHz. Pure inference
 * and end-to-end (quantize -> infer -> dequantize) are measured separately.
 * Results are left in a mailbox read over SWD; the board's UART is not wired
 * to this host.
 */
#include "stm32h7xx_hal.h"
#include "ra_kernel.h"
#include "ra_model.h"
#include "ra_golden.h"

#define BENCH_RUNS      200u
#define MAILBOX_MAGIC   0x48375241u   /* "H7RA" */

#include "status_led.h"

/* ---- results mailbox ---------------------------------------------------- */
typedef struct {
    uint32_t magic;
    uint32_t cpu_hz;
    uint32_t runs;
    uint32_t weight_bytes;
    uint32_t macs;
    uint32_t obs_features;
    uint32_t act_features;
    uint32_t verify_scalar_ok;
    uint32_t verify_simd_ok;
    /* cycles: mean, p50, p99, min, max  for each variant */
    uint32_t scalar_flash[5];
    uint32_t simd_flash[5];
    uint32_t simd_dtcm[5];
    uint32_t e2e_simd_dtcm[5];
    uint32_t dtcm_used;
    int32_t  actions[RA_ACT_FEATURES];
    /* Appended after `actions` so existing readers keep their offsets. */
    uint32_t key_idle_level;   /* PA15 sampled at boot, before any press */
    uint32_t key_presses;      /* debounced press count */
    uint32_t led_level;        /* current green brightness, 0..255 */
} bench_mailbox_t;

volatile bench_mailbox_t g_bench;

/* ---- buffers ------------------------------------------------------------ */
static int16_t obs_q[RA_SCRATCH];
static int16_t act_q[RA_ACT_FEATURES];
#ifdef RA_HIST_FEATURES
/* Branched policy (encoder + concat + actor). ra_kernel walks a flat layer
   list and cannot express the join, so the graph runs as the two chains it
   does support with an int16 splice between them -- the same thing the PL
   CONCAT descriptor does at shift 0. */
static int16_t hist_q[RA_HIST_FEATURES];
static int16_t join_q[RA_JOIN_FEATURES];
static int16_t latent_q[RA_LATENT_DIM];
static ra_layer_t dtcm_enc_layers[RA_ENC_LAYERS];
static ra_layer_t dtcm_act_layers[RA_ACT_LAYERS];
static ra_model_t dtcm_enc, dtcm_act;
#endif
static int16_t scratch[2 * RA_SCRATCH];
static float   obs_f[RA_OBS_FEATURES];
static float   act_f[RA_ACT_FEATURES];
static uint32_t samples[BENCH_RUNS];

/* Weight copy target in DTCM (zero wait state, outside the caches). */
#ifndef RA_DTCM_WEIGHT_BYTES
#define RA_DTCM_WEIGHT_BYTES RA_WEIGHT_BYTES
#endif
#if RA_DTCM_WEIGHT_BYTES <= 118000
static int16_t dtcm_weights[RA_DTCM_WEIGHT_BYTES / 2] __attribute__((section(".dtcm_weights"), aligned(8)));
#ifndef RA_HIST_FEATURES
static ra_layer_t dtcm_layers[RA_LAYERS];
static ra_model_t dtcm_model;
#endif
#define HAVE_DTCM_COPY 1
#else
#define HAVE_DTCM_COPY 0
#endif

/* ---- SysTick ------------------------------------------------------------
 * startup_stm32h723xx.s weak-aliases SysTick_Handler to Default_Handler, whose
 * body is `b .`. HAL_Init() enables the SysTick interrupt, so without this
 * definition the first tick parks the CPU in that loop forever.
 *
 * It bites hard here because HAL_Init() runs before clock_config(): the reload
 * is sized for the 64 MHz HSI and the core then goes to 480 MHz, so SysTick
 * fires every ~133 us while one WS2812 frame inside HAL_SPI_Transmit takes
 * ~120 us. Nearly every LED update overlapped a tick. The symptom was an LED
 * that lit once and went dark while the mailbox still held valid results --
 * the benchmark is pure computation and never calls HAL_GetTick, so it always
 * completed before the hang.
 *
 * HAL_GetTick's timeouts depend on this counter advancing, so the handler must
 * increment it rather than merely return. */
void SysTick_Handler(void)
{
    HAL_IncTick();
}

/* ---- user key: PA15 -----------------------------------------------------
 * Pin and mode come from the board's own example project, not from guesswork:
 *   2D图纸/原理图/CtrBoard-H7_V1.0-240124.pdf   net "USER_KEY(PA15)"
 *   例程/CtrBoard-H7_KEY/User/key_bsp.h         key_1 = GPIOA, GPIO_PIN_15
 *   例程/CtrBoard-H7_KEY/Core/Src/gpio.c        GPIO_MODE_INPUT, GPIO_NOPULL
 * (PA5 is a different net, ADC1_CH18_KEY -- an analog input, not this button.)
 *
 * The pull is external and NOPULL is what the vendor configures, so the idle
 * level is a property of the board, not of the MCU. Rather than assume a
 * polarity, the level sampled at boot is published in the mailbox as
 * key_idle_level and treated as "released"; a press is a sustained departure
 * from it. That is correct for either polarity.
 *
 * PA15 is JTDI. SWD uses only PA13/PA14, so claiming it as GPIO does not
 * disturb the debug link -- which matters here, because a previous build of
 * this firmware locked SWD out of the part. */
#define KEY_PORT        GPIOA
#define KEY_PIN         GPIO_PIN_15
#define KEY_POLL_MS     5u
#define KEY_STABLE_N    4u      /* 4 x 5 ms = 20 ms debounce */

static void key_init(void)
{
    GPIO_InitTypeDef gpio = {0};
    __HAL_RCC_GPIOA_CLK_ENABLE();
    gpio.Pin = KEY_PIN;
    gpio.Mode = GPIO_MODE_INPUT;
    gpio.Pull = GPIO_NOPULL;
    gpio.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(KEY_PORT, &gpio);
}

static inline uint32_t key_raw(void)
{
    return HAL_GPIO_ReadPin(KEY_PORT, KEY_PIN) == GPIO_PIN_SET ? 1u : 0u;
}

/* Perceptually spaced, roughly geometric: equal steps in a linear PWM duty
   look strongly bunched at the bright end. Index 0 is off so the key can also
   turn the LED out. */
static const uint8_t LED_LEVELS[] = { 0u, 4u, 16u, 48u, 120u, 255u };
#define LED_NLEVELS (sizeof LED_LEVELS / sizeof LED_LEVELS[0])

/* ---- clock: the vendor's SystemClock_Config, verbatim ------------------- */
static void panic(void) { for (;;) { } }

static void clock_config(void)
{
    RCC_OscInitTypeDef osc = {0};
    RCC_ClkInitTypeDef clk = {0};

    HAL_PWREx_ConfigSupply(PWR_LDO_SUPPLY);
    __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE0);
    while (!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) { }

    osc.OscillatorType = RCC_OSCILLATORTYPE_HSE;
    osc.HSEState = RCC_HSE_ON;
    osc.PLL.PLLState = RCC_PLL_ON;
    osc.PLL.PLLSource = RCC_PLLSOURCE_HSE;
    osc.PLL.PLLM = 2;
    osc.PLL.PLLN = 40;
    osc.PLL.PLLP = 1;
    osc.PLL.PLLQ = 6;
    osc.PLL.PLLR = 2;
    osc.PLL.PLLRGE = RCC_PLL1VCIRANGE_3;
    osc.PLL.PLLVCOSEL = RCC_PLL1VCOWIDE;
    osc.PLL.PLLFRACN = 0;
    if (HAL_RCC_OscConfig(&osc) != HAL_OK) panic();

    clk.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_PCLK1
                  | RCC_CLOCKTYPE_PCLK2 | RCC_CLOCKTYPE_D3PCLK1 | RCC_CLOCKTYPE_D1PCLK1;
    clk.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
    clk.SYSCLKDivider = RCC_SYSCLK_DIV1;
    clk.AHBCLKDivider = RCC_HCLK_DIV2;
    clk.APB3CLKDivider = RCC_APB3_DIV2;
    clk.APB1CLKDivider = RCC_APB1_DIV2;
    clk.APB2CLKDivider = RCC_APB2_DIV2;
    clk.APB4CLKDivider = RCC_APB4_DIV2;
    if (HAL_RCC_ClockConfig(&clk, FLASH_LATENCY_3) != HAL_OK) panic();
}

/* ---- DWT ---------------------------------------------------------------- */
static void dwt_init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->LAR = 0xC5ACCE55u;          /* unlock, harmless where not required */
    DWT->CYCCNT = 0u;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}
static inline uint32_t dwt_now(void) { return DWT->CYCCNT; }

/* ---- statistics --------------------------------------------------------- */
static void sort_u32(uint32_t *a, uint32_t n)
{
    for (uint32_t i = 1u; i < n; ++i) {
        uint32_t v = a[i]; uint32_t j = i;
        while (j > 0u && a[j - 1u] > v) { a[j] = a[j - 1u]; --j; }
        a[j] = v;
    }
}

static void summarise(uint32_t *a, uint32_t n, volatile uint32_t out[5])
{
    uint64_t sum = 0;
    for (uint32_t i = 0u; i < n; ++i) sum += a[i];
    sort_u32(a, n);
    out[0] = (uint32_t)(sum / n);                 /* mean */
    out[1] = a[n / 2u];                           /* p50  */
    out[2] = a[(uint32_t)((uint64_t)n * 99u / 100u)]; /* p99 */
    out[3] = a[0];                                /* min  */
    out[4] = a[n - 1u];                           /* max  */
}

typedef void (*infer_fn)(const ra_model_t *, const int16_t *, int16_t *, int16_t *);

#ifndef RA_HIST_FEATURES
static void bench(infer_fn fn, const ra_model_t *model, volatile uint32_t out[5])
{
    fn(model, obs_q, act_q, scratch);             /* warm */
    for (uint32_t r = 0u; r < BENCH_RUNS; ++r) {
        uint32_t t0 = dwt_now();
        fn(model, obs_q, act_q, scratch);
        samples[r] = dwt_now() - t0;
    }
    summarise(samples, BENCH_RUNS, out);
}

static void bench_e2e(infer_fn fn, const ra_model_t *model, volatile uint32_t out[5])
{
    for (uint32_t r = 0u; r < BENCH_RUNS; ++r) {
        uint32_t t0 = dwt_now();
        ra_quantize_obs(obs_f, obs_q, RA_OBS_FEATURES);
        fn(model, obs_q, act_q, scratch);
        ra_dequantize_act(act_q, act_f, RA_ACT_FEATURES);
        samples[r] = dwt_now() - t0;
    }
    summarise(samples, BENCH_RUNS, out);
}

static uint32_t verify(infer_fn fn, const ra_model_t *model)
{
    for (uint16_t i = 0u; i < RA_OBS_FEATURES; ++i) obs_q[i] = ra_golden_obs[i];
    fn(model, obs_q, act_q, scratch);
    for (uint16_t i = 0u; i < RA_ACT_FEATURES; ++i)
        if (act_q[i] != ra_golden_act[i]) return 0u;
    return 1u;
}
#else  /* RA_HIST_FEATURES: branched encoder + concat + actor */

static void infer_branched(infer_fn fn, const ra_model_t *enc,
                           const ra_model_t *act, const int16_t *o,
                           const int16_t *h, int16_t *out)
{
    fn(enc, h, latent_q, scratch);
    for (uint16_t i = 0u; i < RA_OBS_FEATURES; ++i) join_q[i] = o[i];
    for (uint16_t i = 0u; i < RA_LATENT_DIM; ++i)
        join_q[RA_OBS_FEATURES + i] = latent_q[i];
    fn(act, join_q, out, scratch);
}

static void bench_branched(infer_fn fn, const ra_model_t *enc,
                           const ra_model_t *act, volatile uint32_t out[5])
{
    infer_branched(fn, enc, act, obs_q, hist_q, act_q);   /* warm */
    for (uint32_t r = 0u; r < BENCH_RUNS; ++r) {
        uint32_t t0 = dwt_now();
        infer_branched(fn, enc, act, obs_q, hist_q, act_q);
        samples[r] = dwt_now() - t0;
    }
    summarise(samples, BENCH_RUNS, out);
}

static void bench_e2e_branched(infer_fn fn, const ra_model_t *enc,
                               const ra_model_t *act, volatile uint32_t out[5])
{
    for (uint32_t r = 0u; r < BENCH_RUNS; ++r) {
        uint32_t t0 = dwt_now();
        ra_quantize_obs(obs_f, obs_q, RA_OBS_FEATURES);
        infer_branched(fn, enc, act, obs_q, hist_q, act_q);
        ra_dequantize_act(act_q, act_f, RA_ACT_FEATURES);
        samples[r] = dwt_now() - t0;
    }
    summarise(samples, BENCH_RUNS, out);
}

static uint32_t verify_branched(infer_fn fn, const ra_model_t *enc,
                                const ra_model_t *act)
{
    /* Every golden vector, not just the first: one vector cannot distinguish
       a correct splice from a lucky one. */
    for (uint32_t v = 0u; v < RA_GOLDEN_COUNT; ++v) {
        const int16_t *o = &ra_golden_obs[v * RA_OBS_FEATURES];
        const int16_t *h = &ra_golden_hist[v * RA_HIST_FEATURES];
        const int16_t *e = &ra_golden_act[v * RA_ACT_FEATURES];
        infer_branched(fn, enc, act, o, h, act_q);
        for (uint16_t i = 0u; i < RA_ACT_FEATURES; ++i)
            if (act_q[i] != e[i]) return 0u;
    }
    return 1u;
}

/* Copy one model's weights and biases into DTCM, advancing the cursor. */
static int16_t *relocate(const ra_model_t *src, ra_layer_t *dst_layers,
                         ra_model_t *dst, int16_t *cursor)
{
    for (uint16_t l = 0u; l < src->n_layers; ++l) {
        const ra_layer_t *sl = &src->layers[l];
        uint32_t wcount = (uint32_t)sl->out_features * sl->row_stride;
        dst_layers[l] = *sl;
        dst_layers[l].weights = cursor;
        for (uint32_t i = 0u; i < wcount; ++i) cursor[i] = sl->weights[i];
        cursor += wcount;
        dst_layers[l].bias = cursor;
        for (uint32_t i = 0u; i < sl->out_features; ++i) cursor[i] = sl->bias[i];
        cursor += sl->out_features;
    }
    *dst = *src;
    dst->layers = dst_layers;
    return cursor;
}
#endif /* RA_HIST_FEATURES */

int main(void)
{
    /* Debug attach window. Without this, firmware that reaches a tight loop
       before the debugger attaches can lock out SWD on a probe whose NRST is
       not wired -- which is exactly how the first build bricked access to this
       board. Spin at the reset clock for a moment so a hot-plug connect always
       has somewhere to land. */
    for (volatile uint32_t i = 0; i < 4000000u; ++i) { __NOP(); }

    /* Keep the debug unit alive in low-power states. */
    DBGMCU->CR |= DBGMCU_CR_DBG_SLEEPD1 | DBGMCU_CR_DBG_STOPD1 | DBGMCU_CR_DBG_STANDBYD1;
    SCB_EnableICache();
    SCB_EnableDCache();
    HAL_Init();
    clock_config();
    SystemCoreClockUpdate();
    dwt_init();
    ra_tables_init();
    /* LED3 (WS2812 on PA7/SPI6_MOSI). Every status_led call blocks for
       hundreds of microseconds, so all of them sit at phase boundaries --
       never between a dwt_now() pair. Blue means "measuring". */
    status_led_init();
    status_led_set(0u, 0u, 24u);

    g_bench.magic = 0u;
    g_bench.cpu_hz = SystemCoreClock;
    g_bench.runs = BENCH_RUNS;
    g_bench.weight_bytes = RA_WEIGHT_BYTES;
    g_bench.macs = RA_MACS;
    g_bench.obs_features = RA_OBS_FEATURES;
    g_bench.act_features = RA_ACT_FEATURES;

    for (uint16_t i = 0u; i < RA_OBS_FEATURES; ++i) {
        obs_q[i] = ra_golden_obs[i];
        obs_f[i] = (float)ra_golden_obs[i] * (1.0f / 256.0f);
    }
    for (uint16_t i = RA_OBS_FEATURES; i < RA_SCRATCH; ++i) obs_q[i] = 0;
#ifdef RA_HIST_FEATURES
    for (uint16_t i = 0u; i < RA_HIST_FEATURES; ++i) hist_q[i] = ra_golden_hist[i];

    g_bench.verify_scalar_ok =
        verify_branched(ra_infer_scalar, &ra_model_enc, &ra_model_act);
    g_bench.verify_simd_ok =
        verify_branched(ra_infer_simd, &ra_model_enc, &ra_model_act);

    bench_branched(ra_infer_scalar, &ra_model_enc, &ra_model_act, g_bench.scalar_flash);
    bench_branched(ra_infer_simd,   &ra_model_enc, &ra_model_act, g_bench.simd_flash);

#if HAVE_DTCM_COPY
    {
        int16_t *cursor = dtcm_weights;
        cursor = relocate(&ra_model_enc, dtcm_enc_layers, &dtcm_enc, cursor);
        cursor = relocate(&ra_model_act, dtcm_act_layers, &dtcm_act, cursor);
        g_bench.dtcm_used = (uint32_t)((cursor - dtcm_weights) * 2);
        bench_branched(ra_infer_simd, &dtcm_enc, &dtcm_act, g_bench.simd_dtcm);
        bench_e2e_branched(ra_infer_simd, &dtcm_enc, &dtcm_act, g_bench.e2e_simd_dtcm);
    }
#else
    g_bench.dtcm_used = 0u;
    bench_e2e_branched(ra_infer_simd, &ra_model_enc, &ra_model_act,
                       g_bench.e2e_simd_dtcm);
#endif
#else  /* single feed-forward chain */
    g_bench.verify_scalar_ok = verify(ra_infer_scalar, &ra_model);
    g_bench.verify_simd_ok   = verify(ra_infer_simd,   &ra_model);

    bench(ra_infer_scalar, &ra_model, g_bench.scalar_flash);
    bench(ra_infer_simd,   &ra_model, g_bench.simd_flash);

#if HAVE_DTCM_COPY
    {
        /* Relocate every weight and bias array into DTCM and rebuild the
           descriptors so the SIMD kernel reads zero-wait-state memory. */
        int16_t *cursor = dtcm_weights;
        for (uint16_t l = 0u; l < RA_LAYERS; ++l) {
            const ra_layer_t *src = &ra_model.layers[l];
            uint32_t wcount = (uint32_t)src->out_features * src->row_stride;
            dtcm_layers[l] = *src;
            dtcm_layers[l].weights = cursor;
            for (uint32_t i = 0u; i < wcount; ++i) cursor[i] = src->weights[i];
            cursor += wcount;
            dtcm_layers[l].bias = cursor;
            for (uint32_t i = 0u; i < src->out_features; ++i) cursor[i] = src->bias[i];
            cursor += src->out_features;
        }
        g_bench.dtcm_used = (uint32_t)((cursor - dtcm_weights) * 2);
        dtcm_model = ra_model;
        dtcm_model.layers = dtcm_layers;
        bench(ra_infer_simd, &dtcm_model, g_bench.simd_dtcm);
        bench_e2e(ra_infer_simd, &dtcm_model, g_bench.e2e_simd_dtcm);
    }
#else
    g_bench.dtcm_used = 0u;
    bench_e2e(ra_infer_simd, &ra_model, g_bench.e2e_simd_dtcm);
#endif

#endif /* RA_HIST_FEATURES */

    for (uint16_t i = 0u; i < RA_ACT_FEATURES; ++i) g_bench.actions[i] = act_q[i];
    __DMB();
    g_bench.magic = MAILBOX_MAGIC;

    /* All measurement is finished and published; the LED cannot perturb it
       from here. Green = both numerics checks matched the golden vector,
       red = they did not. */
    const uint8_t pass = g_bench.verify_scalar_ok && g_bench.verify_simd_ok;

    /* Busy loop rather than WFI: the debugger must be able to hot-plug attach
       and read this mailbox without resetting the target. The loop only reads
       CYCCNT -- it never resets it.

       The LED is steady, not blinking: status_led_set is written only when the
       level actually changes, so the WS2812 holds its last latched colour and
       there is no periodic SPI traffic. Green = both numerics checks matched
       the golden vectors, red = they did not; the key sets the brightness. */
    key_init();
    const uint32_t idle = key_raw();
    g_bench.key_idle_level = idle;

    uint32_t level_idx = 3u;                 /* 48/255, a visible default */
    uint32_t presses = 0u;
    uint32_t stable = 0u;
    uint32_t last_sample = idle;
    uint32_t held = 0u;                      /* debounced state: 1 = pressed */
    uint32_t mark = dwt_now();
    const uint32_t poll_cycles = (uint32_t)(SystemCoreClock / 1000u) * KEY_POLL_MS;

    g_bench.key_presses = 0u;
    g_bench.led_level = LED_LEVELS[level_idx];
    status_led_set(pass ? 0u : LED_LEVELS[level_idx],
                   pass ? LED_LEVELS[level_idx] : 0u, 0u);

    for (;;) {
        if ((dwt_now() - mark) < poll_cycles) { __NOP(); continue; }
        mark = dwt_now();

        const uint32_t now = key_raw();
        stable = (now == last_sample) ? (stable + 1u) : 0u;
        last_sample = now;
        if (stable < KEY_STABLE_N) continue;

        /* "Pressed" is any level other than the one seen at boot, so this
           works whether the board pulls the net up or down. */
        const uint32_t down = (now != idle) ? 1u : 0u;
        if (down && !held) {                 /* rising edge of a press */
            level_idx = (level_idx + 1u) % LED_NLEVELS;
            presses++;
            g_bench.key_presses = presses;
            g_bench.led_level = LED_LEVELS[level_idx];
            status_led_set(pass ? 0u : LED_LEVELS[level_idx],
                           pass ? LED_LEVELS[level_idx] : 0u, 0u);
        }
        held = down;
    }
}
