#include <config.h>
#include <extract_codebook_index_fast.h>

int extract_codebook_index_fast(const int8_t *buffer)
{

  int max_index = 0;
  int8_t max_value = buffer[0];

#ifdef ENABLE_UNROLL_EXTRACT
#pragma GCC unroll 64
#endif
  for (int i = 0; i < OUTPUT_FEATURE_SIZE; i++)
  {
    if (buffer[i] > max_value)
    {
      max_value = buffer[i];
      max_index = i;
    }
  }

  return max_index + 1;
}