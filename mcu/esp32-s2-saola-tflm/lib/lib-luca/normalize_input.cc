#include <config.h>
#include <normalize_input.h>

void normalize_input(float *buffer_in, float *buffer_out)
{

#ifdef ENABLE_UNROLL_NORMALIZE
#pragma GCC unroll 1024
#endif
  for (int i = 0; i < INPUT_FEATURE_SIZE; ++i)
  {
    // buffer_out[i] = (buffer_in[i] - mean_array[i]) / sqrtf(variance_array[i]);
    //buffer_out[i] = (buffer_in[i] - mean_array[i]) * variance_array_sqrt_inv[i];
    buffer_out[i] = (buffer_in[i] - mean_array);
  }
}
