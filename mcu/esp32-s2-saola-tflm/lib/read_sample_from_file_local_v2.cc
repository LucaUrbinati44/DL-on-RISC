#include "tensorflow/lite/micro/examples/ml_on_risc/config.h"
#include "tensorflow/lite/micro/examples/ml_on_risc/read_sample_from_file_local_v2.h"

bool read_sample_from_file_local_v2(float* buffer) {
    static size_t sample_idx = 0;

    if (sample_idx >= NUM_SAMPLES) {
        return false;  // fine dati
    }

    for (size_t i = 0; i < INPUT_FEATURE_SIZE; i++) {
        buffer[i] = samples[sample_idx][i];
    }

    sample_idx++;
    return true;
}