#include "tensorflow/lite/micro/examples/ml_on_risc/config.h"
#include "tensorflow/lite/micro/examples/ml_on_risc/extract_codebook_index.h"

int extract_codebook_index(float* buffer) {

  int max_index = 0;
  float max_value = buffer[0];

  for (int i = 1; i < OUTPUT_FEATURE_SIZE; i++) {
      if (buffer[i] > max_value) {
          max_value = buffer[i];
          max_index = i;
          //DPRINTF("Aggiorno massimo\n");
      }
  }

  return max_index; // E' già 1-based perchè ho usato i=1 come inizializzazione
}