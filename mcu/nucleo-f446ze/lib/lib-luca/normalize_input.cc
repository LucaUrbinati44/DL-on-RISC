#include <config.h>
#include <normalize_input.h>

// #ifdef NORMALIZE_FROM_RAM
// float *variance_ram = nullptr;
// float *mean_ram = nullptr;
// #endif

void normalize_input(float *buffer_in, float *buffer_out)
{
  // #ifdef NORMALIZE_FROM_RAM
  //   float *mean = mean_ram;
  //   float *var = variance_ram;
  // #else
  //   const float *mean = mean_array;
  //   const float *var = variance_array;
  // #endif

#ifdef ENABLE_UNROLL_NORMALIZE
#pragma GCC unroll 1024
#endif
  for (int i = 0; i < INPUT_FEATURE_SIZE; ++i)
  {
    // buffer_out[i] = (buffer_in[i] - mean[i]) / sqrtf(var[i]);
    // buffer_out[i] = (buffer_in[i] - mean_array[i]) / sqrtf(variance_array[i]);
    buffer_out[i] = (buffer_in[i] - mean_array[i]) * variance_array_sqrt_inv[i];
  }
}
