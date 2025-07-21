#include "tensorflow/lite/micro/examples/ml_on_risc/read_sample_from_uart_local.h"

// Percorso fisso al file
const std::string kSampleFilePath = "/mnt/c/Users/Work/Desktop/deepMIMO/RIS/renode_boards/litex-vexriscv-tensorflow-lite-demo/renode/test_set_small.data";
static std::ifstream sample_file;
static bool file_loaded = false;

bool read_sample_from_uart_local(float* buffer) {
  if (!file_loaded) {
    sample_file.open(kSampleFilePath);
    if (!sample_file.is_open()) {
      std::cerr << "Impossibile aprire il file: " << kSampleFilePath << std::endl;
      return false;
    }
    file_loaded = true;
  }

  std::string line;
  if (!std::getline(sample_file, line)) {
    std::cerr << "Fine del file raggiunta o errore di lettura." << std::endl;
    return false;
  }

  std::istringstream iss(line);
  for (int i = 0; i < FEATURE_SIZE; ++i) {
    if (!(iss >> buffer[i])) {
      std::cerr << "Errore: meno di " << FEATURE_SIZE << " feature nella riga corrente." << std::endl;
      return false;
    }
  }

  return true;
}




