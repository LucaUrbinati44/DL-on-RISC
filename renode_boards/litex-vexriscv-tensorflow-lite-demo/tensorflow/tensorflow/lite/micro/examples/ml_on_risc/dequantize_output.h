#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_ML_ON_RISC_DEQUANTIZE_OUTPUT_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_ML_ON_RISC_DEQUANTIZE_OUTPUT_H_

#include "config.h"

void dequantize_output(int8_t* buffer_in, float scale, int zero_point, float* buffer_out);

#endif  // TENSORFLOW_LITE_MICRO_EXAMPLES_ML_ON_RISC_DEQUANTIZE_OUTPUT_H_