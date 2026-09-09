/* Host-side check that the generated H7 model reproduces the Python reference.
 * Both kernels must agree with ra_golden.h bit for bit; the SMLALD path is
 * emulated on x86 with the exact semantics of the instruction. */
#include <stdio.h>
#include <string.h>
#include "ra_kernel.h"
#include "ra_probe.h"
#include "ra_model.h"
#include "ra_golden.h"

static int16_t scratch[2 * RA_SCRATCH];
static int16_t join[RA_JOIN_FEATURES];

/* Entry i of the ELU ROM, probed through the public API: at x = -8i the
   interpolation remainder is zero, so ra_elu returns table[i] verbatim.
   This is the regression guard for the Taylor-series cancellation bug that
   made entries 247..255 come out at -257 instead of -256. */
static const int16_t expected_rom[256] = {
    0, -8, -16, -23, -30, -37, -44, -50, -57, -63, -69, -74, -80, -85, -91, -96,
    -101, -106, -110, -115, -119, -123, -127, -131, -135, -139, -142, -146, -149, -153, -156, -159,
    -162, -165, -168, -170, -173, -175, -178, -180, -183, -185, -187, -189, -191, -193, -195, -197,
    -199, -201, -202, -204, -206, -207, -209, -210, -212, -213, -214, -215, -217, -218, -219, -220,
    -221, -222, -223, -224, -225, -226, -227, -228, -229, -230, -231, -231, -232, -233, -234, -234,
    -235, -236, -236, -237, -237, -238, -239, -239, -240, -240, -241, -241, -242, -242, -242, -243,
    -243, -244, -244, -244, -245, -245, -245, -246, -246, -246, -247, -247, -247, -248, -248, -248,
    -248, -249, -249, -249, -249, -249, -250, -250, -250, -250, -250, -251, -251, -251, -251, -251,
    -251, -251, -252, -252, -252, -252, -252, -252, -252, -252, -253, -253, -253, -253, -253, -253,
    -253, -253, -253, -253, -253, -254, -254, -254, -254, -254, -254, -254, -254, -254, -254, -254,
    -254, -254, -254, -254, -254, -255, -255, -255, -255, -255, -255, -255, -255, -255, -255, -255,
    -255, -255, -255, -255, -255, -255, -255, -255, -255, -255, -255, -255, -255, -255, -255, -255,
    -255, -255, -255, -255, -255, -255, -255, -255, -256, -256, -256, -256, -256, -256, -256, -256,
    -256, -256, -256, -256, -256, -256, -256, -256, -256, -256, -256, -256, -256, -256, -256, -256,
    -256, -256, -256, -256, -256, -256, -256, -256, -256, -256, -256, -256, -256, -256, -256, -256,
    -256, -256, -256, -256, -256, -256, -256, -256, -256, -256, -256, -256, -256, -256, -256, -256,
};

static int check_rom(void)
{
    int bad = 0;
    for (int i = 0; i < 256; ++i) {
        int16_t got = ra_elu_probe((int16_t)(-8 * i));
        if (got != expected_rom[i]) {
            bad++;
            if (bad <= 8) printf("  rom[%3d] got %d want %d\n", i, got, expected_rom[i]);
        }
    }
    printf("ELU ROM entries differing from the RTL: %d/256\n", bad);
    return bad;
}

int main(void)
{
    ra_tables_init();
    int bad_rom = check_rom();
    int bad_simd = 0, bad_scalar = 0;
    for (int v = 0; v < RA_GOLDEN_COUNT; ++v) {
        const int16_t *o = &ra_golden_obs[(size_t)v * RA_OBS_FEATURES];
        const int16_t *h = &ra_golden_hist[(size_t)v * RA_HIST_FEATURES];
        const int16_t *e = &ra_golden_act[(size_t)v * RA_ACT_FEATURES];
        int16_t a[RA_ACT_FEATURES], lat[RA_LATENT_DIM];

        ra_infer_wheelleg(o, h, a, join, scratch);
        for (int i = 0; i < RA_ACT_FEATURES; ++i)
            if (a[i] != e[i]) {
                bad_simd++;
                printf("  simd   v%d[%d] got %d want %d\n", v, i, a[i], e[i]);
            }

        ra_infer_scalar(&ra_model_enc, h, lat, scratch);
        memcpy(join, o, RA_OBS_FEATURES * sizeof(int16_t));
        memcpy(join + RA_OBS_FEATURES, lat, RA_LATENT_DIM * sizeof(int16_t));
        ra_infer_scalar(&ra_model_act, join, a, scratch);
        for (int i = 0; i < RA_ACT_FEATURES; ++i)
            if (a[i] != e[i]) {
                bad_scalar++;
                printf("  scalar v%d[%d] got %d want %d\n", v, i, a[i], e[i]);
            }
    }
    printf("vectors %d  simd mismatches %d  scalar mismatches %d\n",
           RA_GOLDEN_COUNT, bad_simd, bad_scalar);
    printf("H7_HOST_CHECK: %s\n",
           (bad_rom || bad_simd || bad_scalar) ? "FAIL" : "PASS");
    return (bad_rom || bad_simd || bad_scalar) ? 1 : 0;
}
