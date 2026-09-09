#ifndef STATUS_LED_H
#define STATUS_LED_H

#include "stm32h7xx_hal.h"

/* LED3, the WS2812 on PA7 / SPI6_MOSI. See status_led.c for the pin evidence.
 *
 * Both calls block for hundreds of microseconds. Never call either one inside
 * a benchmark timing region. */
void status_led_init(void);
void status_led_set(uint8_t r, uint8_t g, uint8_t b);

#endif
