#include <Arduino.h>
#include <stdint.h>

#define DEBUG_QUANTIZE
#define DEBUG_DEQUANTIZE

#define MCU_RAM_BYTES 320*1024
#define BAUD_RATE 921600
#define CHUNK_SIZE_MAX 3584

#define INPUT_FEATURE_SIZE 3584
#define OUTPUT_FEATURE_SIZE 4096

#define TENSOR_ARENA_KB 57

const float mean_array = 0.00017636189295444638;
