#include <Arduino.h>
//#include "system_stm32f4xx.h"
#include <stdint.h>

#define DEBUG_QUANTIZE
#define DEBUG_DEQUANTIZE

#define MCU_RAM_BYTES 128*1024

#define BAUD_RATE 115200
#define CHUNK_SIZE_MAX 3584

#define INPUT_FEATURE_SIZE 3584
#define OUTPUT_FEATURE_SIZE 4096

#define TENSOR_ARENA_KB 100

const float mean_array = 0.00017636189295444638;
