/* Status LED for the H7 benchmark.
 *
 * The only indicator on this board that can show green is LED3, a WS2812
 * addressable RGB whose DIN is driven by SPI6_MOSI on PA7. Verified from the
 * CtrBoard-H7 schematic (LED3 = DIN/VDD/VSS/DOUT, DIN net SPI6_MOSI(PA07)) and
 * from the vendor's own CtrBoard-H7_WS2812 example, whose HAL_SPI_MspInit
 * states "PA7 ------> SPI6_MOSI" with AF8.
 *
 * The other three LEDs on the board (LED1/LED2/LED5) are all LED-R 0603 and
 * sit on the drains of Q4/Q5/Q6, the power-output switches behind the OUT_EN /
 * 5V_EN pins. Lighting one means energising a power rail on a motor control
 * board, so they are not usable as a status indicator and are left alone.
 *
 * A WS2812 has no "active level" -- it is an NRZ data protocol, not a driven
 * pin. Colour is sent as 24 bits in GRB order, each WS2812 bit expanded to one
 * SPI byte so the SPI shift register generates the pulse widths. SPI6 is
 * clocked from HSE (24 MHz) with prescaler 4 = 6 MHz, i.e. 166.7 ns per SPI
 * bit, so one WS2812 bit takes 1.33 us:
 *
 *     0 -> 0x60 = 0b01100000 : 333 ns high  (T0H spec 350 +/- 150 ns)
 *     1 -> 0x78 = 0b01111000 : 667 ns high  (T1H spec 700 +/- 150 ns)
 *
 * Byte values and bit order are taken from the vendor driver rather than
 * derived, so the timing matches hardware that is known to work on this board.
 *
 * Only PA7 is configured. The vendor example also sets up PA5 as SPI6_SCK, but
 * the LED does not use it and PA5 is ADC1_CH18 / KEY on this board, so leaving
 * it untouched keeps this strictly non-intrusive.
 *
 * TIMING CONTRACT: every entry point here is blocking and takes hundreds of
 * microseconds. Nothing in this file may be called between a dwt_now() pair.
 * status_led_set() is called only at phase boundaries and the heartbeat runs
 * only in the terminal idle loop, after the mailbox has been published.
 */
#include "status_led.h"

#define WS2812_BIT0 0x60u
#define WS2812_BIT1 0x78u

static SPI_HandleTypeDef hspi6;
static uint8_t led_ok;

void status_led_init(void)
{
    RCC_PeriphCLKInitTypeDef pclk = {0};
    GPIO_InitTypeDef gpio = {0};

    pclk.PeriphClockSelection = RCC_PERIPHCLK_SPI6;
    pclk.Spi6ClockSelection = RCC_SPI6CLKSOURCE_HSE;
    if (HAL_RCCEx_PeriphCLKConfig(&pclk) != HAL_OK) return;

    __HAL_RCC_SPI6_CLK_ENABLE();
    __HAL_RCC_GPIOA_CLK_ENABLE();

    gpio.Pin = GPIO_PIN_7;                    /* MOSI only; PA5 is ADC/KEY */
    gpio.Mode = GPIO_MODE_AF_PP;
    gpio.Pull = GPIO_NOPULL;
    gpio.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    gpio.Alternate = GPIO_AF8_SPI6;
    HAL_GPIO_Init(GPIOA, &gpio);

    hspi6.Instance = SPI6;
    hspi6.Init.Mode = SPI_MODE_MASTER;
    hspi6.Init.Direction = SPI_DIRECTION_2LINES_TXONLY;
    hspi6.Init.DataSize = SPI_DATASIZE_8BIT;
    hspi6.Init.CLKPolarity = SPI_POLARITY_LOW;
    hspi6.Init.CLKPhase = SPI_PHASE_2EDGE;
    hspi6.Init.NSS = SPI_NSS_SOFT;
    hspi6.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_4;
    hspi6.Init.FirstBit = SPI_FIRSTBIT_MSB;
    hspi6.Init.TIMode = SPI_TIMODE_DISABLE;
    hspi6.Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
    hspi6.Init.NSSPMode = SPI_NSS_PULSE_ENABLE;
    hspi6.Init.NSSPolarity = SPI_NSS_POLARITY_LOW;
    hspi6.Init.FifoThreshold = SPI_FIFO_THRESHOLD_01DATA;
    hspi6.Init.MasterSSIdleness = SPI_MASTER_SS_IDLENESS_00CYCLE;
    hspi6.Init.MasterInterDataIdleness = SPI_MASTER_INTERDATA_IDLENESS_00CYCLE;
    hspi6.Init.MasterReceiverAutoSusp = SPI_MASTER_RX_AUTOSUSP_DISABLE;
    hspi6.Init.MasterKeepIOState = SPI_MASTER_KEEP_IO_STATE_DISABLE;
    hspi6.Init.IOSwap = SPI_IO_SWAP_DISABLE;
    led_ok = (HAL_SPI_Init(&hspi6) == HAL_OK) ? 1u : 0u;
}

void status_led_set(uint8_t r, uint8_t g, uint8_t b)
{
    uint8_t tx[24];
    uint8_t zero[64] = {0};

    if (!led_ok) return;

    for (int i = 0; i < 8; ++i) {          /* GRB, MSB first */
        tx[7 - i]  = ((g >> i) & 1u) ? WS2812_BIT1 : WS2812_BIT0;
        tx[15 - i] = ((r >> i) & 1u) ? WS2812_BIT1 : WS2812_BIT0;
        tx[23 - i] = ((b >> i) & 1u) ? WS2812_BIT1 : WS2812_BIT0;
    }
    HAL_SPI_Transmit(&hspi6, tx, sizeof tx, 100u);
    /* Latch: >50 us idle low. 64 zero bytes at 6 MHz is 85 us. */
    HAL_SPI_Transmit(&hspi6, zero, sizeof zero, 100u);
}
