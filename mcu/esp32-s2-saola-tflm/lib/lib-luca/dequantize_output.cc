#include <config.h>
#include <dequantize_output.h>

void dequantize_output(int8_t *buffer_in, float scale, int zero_point, float *buffer_out)
{
#ifdef ENABLE_UNROLL_DEQUANTIZE
#pragma GCC unroll 64
#endif
  for (int i = 0; i < OUTPUT_FEATURE_SIZE; ++i)
  {
    buffer_out[i] = static_cast<float>((static_cast<int>(buffer_in[i]) - zero_point) * scale);
  }
}