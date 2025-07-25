#include "tensorflow/lite/micro/examples/ml_on_risc/read_sample_from_file.h"

/*
bool read_sample_from_file(float* buffer) {
    float value;
    for (int i = 0; i < INPUT_FEATURE_SIZE; i++) {
        if (scanf("%f", &value) != 1) {
            return false;  // fine dati o errore
        }
        buffer[i] = value;
        DPRINTF("%f ", (double)value);
    }
    DPRINTF("\n");
    return true;
}
*/

int input[FEATURE_SIZE];

void read_sample_from_file() {
    char buffer[8192]; // abbastanza per una riga lunga
    fgets(buffer, sizeof(buffer), stdin);  // stdin è mappato alla UART in Zephyr

    char *token = strtok(buffer, " ");
    for (int i = 0; i < INPUT_FEATURE_SIZE && token != NULL; ++i) {
        input[i] = atoi(token);
        token = strtok(NULL, " ");
    }
}