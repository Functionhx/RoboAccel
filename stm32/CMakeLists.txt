cmake_minimum_required(VERSION 3.22)
project(h7_bench C ASM)
set(CMAKE_C_STANDARD 11)

# Which policy to build. -DRA_MODEL=go2|g1|go2_orig|g1_orig
set(RA_MODEL "go2" CACHE STRING "model directory under models/")

set(REF   "/home/as/vllm/fpga/st-h7/ref")
set(BOARD "${REF}/dm-mc02/例程/CtrBoard-H7_ALL")
set(HAL "${BOARD}/Drivers/STM32H7xx_HAL_Driver/Src")

add_executable(${PROJECT_NAME}
    src/main.c
    src/ra_kernel.c
    src/status_led.c
    src/system_stm32h7xx.c
    src/startup_stm32h723xx.s
    models/${RA_MODEL}/ra_model.c
    ${HAL}/stm32h7xx_hal.c
    ${HAL}/stm32h7xx_hal_rcc.c
    ${HAL}/stm32h7xx_hal_rcc_ex.c
    ${HAL}/stm32h7xx_hal_pwr.c
    ${HAL}/stm32h7xx_hal_pwr_ex.c
    ${HAL}/stm32h7xx_hal_cortex.c
    ${HAL}/stm32h7xx_hal_flash.c
    ${HAL}/stm32h7xx_hal_flash_ex.c
    ${HAL}/stm32h7xx_hal_gpio.c
    ${HAL}/stm32h7xx_hal_spi.c
    ${HAL}/stm32h7xx_hal_spi_ex.c
    ${HAL}/stm32h7xx_hal_dma.c
    ${HAL}/stm32h7xx_hal_dma_ex.c
    ${HAL}/stm32h7xx_hal_mdma.c
    ${HAL}/stm32h7xx_hal_exti.c
)

target_include_directories(${PROJECT_NAME} PRIVATE
    src
    models/${RA_MODEL}
    ${REF}/cmsis_device_h7/Include
    ${BOARD}/Drivers/CMSIS/Include
    ${BOARD}/Drivers/STM32H7xx_HAL_Driver/Inc
)

target_compile_definitions(${PROJECT_NAME} PRIVATE STM32H723xx USE_HAL_DRIVER USE_PWR_LDO_SUPPLY)

# -O3 and full M7 features: do not handicap the MCU.
target_compile_options(${PROJECT_NAME} PRIVATE
    -mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard
    -O3 -funroll-loops -ffunction-sections -fdata-sections
    -Wall -Wextra -Wno-unused-parameter
    $<$<COMPILE_LANGUAGE:C>:-std=gnu11>
)

target_link_options(${PROJECT_NAME} PRIVATE
    -mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard
    -T${CMAKE_SOURCE_DIR}/ld/stm32h723vg_flash.ld
    --specs=nano.specs --specs=nosys.specs
    -Wl,--gc-sections -Wl,-Map=${PROJECT_NAME}.map,--cref
    -Wl,--print-memory-usage
)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_SIZE} $<TARGET_FILE:${PROJECT_NAME}>
    COMMAND ${CMAKE_OBJCOPY} -O binary $<TARGET_FILE:${PROJECT_NAME}> ${PROJECT_NAME}.bin
    COMMAND ${CMAKE_OBJCOPY} -O ihex   $<TARGET_FILE:${PROJECT_NAME}> ${PROJECT_NAME}.hex
)
