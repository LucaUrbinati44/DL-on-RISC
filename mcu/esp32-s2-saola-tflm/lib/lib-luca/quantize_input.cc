#include <config.h>
#include <quantize_input.h>

void quantize_input(float *buffer_in, float scale_inv, int zero_point, int8_t *buffer_out)
{

#ifdef ENABLE_UNROLL_QUANTIZE
#pragma GCC unroll 128
#endif
  for (int i = 0; i < INPUT_FEATURE_SIZE; ++i)
  {

    // int32_t q = static_cast<int32_t>(roundf(buffer_in[i] / scale)) + zero_point;
    int32_t q = static_cast<int32_t>(roundf(buffer_in[i] * scale_inv)) + zero_point;
    if (q > 127)
      q = 127;
    if (q < -128)
      q = -128;

    buffer_out[i] = static_cast<int8_t>(q);
  }
}
