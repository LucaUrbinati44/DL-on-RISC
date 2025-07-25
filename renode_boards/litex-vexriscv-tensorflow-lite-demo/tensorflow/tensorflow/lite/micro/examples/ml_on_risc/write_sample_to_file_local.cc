#include "tensorflow/lite/micro/examples/ml_on_risc/write_sample_to_file_local.h"

// Percorso fisso al file di output
const std::string kOutputFilePath = "/mnt/c/Users/Work/Desktop/deepMIMO/RIS/renode_boards/litex-vexriscv-tensorflow-lite-demo/renode/output_codebook_small.data";
static std::ofstream output_file;
static bool file_initialized = false;

void write_sample_to_file_local(int value) {
  if (!file_initialized) {
    output_file.open(kOutputFilePath);
    if (!output_file.is_open()) {
      std::cerr << "Impossibile aprire il file in scrittura: " << kOutputFilePath << std::endl;
      return;
    }
    file_initialized = true;
  }

  output_file << value;
  output_file << '\n';
  output_file.flush();
}