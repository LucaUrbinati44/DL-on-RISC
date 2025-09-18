#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_ML_ON_RISC_NORMALIZE_INPUT_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_ML_ON_RISC_NORMALIZE_INPUT_H_

// #ifdef NORMALIZE_FROM_RAM
// extern float *variance_ram;
// extern float *mean_ram;
// #endif

void normalize_input(float *buffer_in, float *buffer_out);

#endif // TENSORFLOW_LITE_MICRO_EXAMPLES_ML_ON_RISC_NORMALIZE_INPUT_H_