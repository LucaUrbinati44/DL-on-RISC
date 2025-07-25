#include "tensorflow/lite/micro/examples/ml_on_risc/dequantize_output.h"

void dequantize_output(int8_t* buffer_in, float scale, int zero_point, float* buffer_out) {

  DPRINTF("quantized_output: ");
  for (int i = 0; i < OUTPUT_FEATURE_SIZE; ++i) {
    DPRINTF("%d ", (int)buffer_in[i]);
  }
  DPRINTF("\n\n");

  DPRINTF("dequantized_output: ");
  for (int i = 0; i < OUTPUT_FEATURE_SIZE; ++i) {
    buffer_out[i] = static_cast<float>((static_cast<int>(buffer_in[i]) - zero_point) * scale);
    DPRINTF("%0.6f ", (double)buffer_out[i]);
  }
  DPRINTF("\n\n");

}