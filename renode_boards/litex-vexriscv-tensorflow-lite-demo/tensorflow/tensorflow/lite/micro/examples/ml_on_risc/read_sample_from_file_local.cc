#include "tensorflow/lite/micro/examples/ml_on_risc/read_sample_from_file_local.h"

// Percorso fisso al file
const std::string kInputFilePath = "/mnt/c/Users/Work/Desktop/deepMIMO/RIS/renode_boards/litex-vexriscv-tensorflow-lite-demo/renode/test_set_small.data";
static std::ifstream input_file;
static bool file_loaded = false;

bool read_sample_from_file_local(float* buffer) {

  // Apri file una volta sola
  if (!file_loaded) {
    input_file.open(kInputFilePath);
    if (!input_file.is_open()) {
      std::cerr << "Impossibile aprire il file: " << kInputFilePath << std::endl;
      return false;
    }
    file_loaded = true;
  }

  // Leggi una riga
  std::string line;
  if (!std::getline(input_file, line)) {
    std::cerr << "Fine del file raggiunta o errore di lettura." << std::endl;
    return false;
  }

  // Leggi ogni elemento di una riga
  DPRINTF("float_input: ");
  std::istringstream iss(line);
  for (int i = 0; i < INPUT_FEATURE_SIZE; ++i) {
    if (!(iss >> buffer[i])) {
      std::cerr << "Errore: meno di " << INPUT_FEATURE_SIZE << " feature nella riga corrente." << std::endl;
      return false;
    }
    DPRINTF("%0.6f ", (double)buffer[i]);
  }
  DPRINTF("\n\n");

  return true;
}