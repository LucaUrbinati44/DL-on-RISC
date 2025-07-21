#include "tensorflow/lite/micro/examples/ml_on_risc/read_sample_from_uart.h"

bool read_sample_from_uart(float* buffer) {
    for (int i = 0; i < FEATURE_SIZE; i++) {
        if (scanf("%f", &buffer[i]) != 1) {
            return false;  // fine dati o errore
        }
        printf("%f ", (double)buffer[i]);
    }
    printf("\n");
    return true;
}