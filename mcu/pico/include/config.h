#include <Arduino.h>
#include <stdint.h>

#define DEBUG_QUANTIZE
#define DEBUG_DEQUANTIZE

#define BAUD_RATE 115200
#define CHUNK_SIZE_MAX 128

#define INPUT_FEATURE_SIZE 1024
#define OUTPUT_FEATURE_SIZE 1024

const float mean_array = -5.190452156966785e-06;
