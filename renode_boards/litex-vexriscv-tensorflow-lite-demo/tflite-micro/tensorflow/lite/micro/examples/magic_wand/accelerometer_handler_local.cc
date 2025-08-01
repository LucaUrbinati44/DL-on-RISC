#include "tensorflow/lite/micro/examples/magic_wand/accelerometer_handler_local.h"

#include <cstdio>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

#define BUFLEN 300
int begin_index = 0;
float bufx[BUFLEN] = {0.0f};
float bufy[BUFLEN] = {0.0f};
float bufz[BUFLEN] = {0.0f};
bool initial = true;
int flush_buffer_counter = 0;

// Buffer temporaneo per tutti i dati caricati
std::vector<std::vector<float>> all_samples;
size_t all_samples_index = 0;

// Funzione di utilità per caricare dati da file
void load_data_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        printf("Errore: impossibile aprire il file %s\n", filename.c_str());
        return;
    }
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        float x, y, z;
        //if (!(iss >> x >> y >> z)) continue;
        if (!(iss >> x >> y >> z)) {
            printf("Riga non valida: '%s'\n", line.c_str());
            continue;
        }
        all_samples.push_back({x, y, z});
        //printf("%d, %d, %d\n", (int)x, (int)y, (int)z);
    }
}

// Funzione di utilità per aggiungere sample costanti N volte
void feed_sample(float x, float y, float z, int n) {
    for (int i = 0; i < n; ++i) {
        all_samples.push_back({x, y, z});
        //printf("%d, %d, %d\n", (int)x, (int)y, (int)z);
    }
}

TfLiteStatus SetupAccelerometer(tflite::ErrorReporter* error_reporter) {
    all_samples.clear();
    all_samples_index = 0;
    begin_index = 0;
    initial = true;
    flush_buffer_counter = 0;

    // Mima la sequenza del .resc
    load_data_file("/mnt/c/Users/Work/Desktop/deepMIMO/RIS/renode_examples/litex-vexriscv-tensorflow-lite-demo/renode/circle.data");
    feed_sample(0.0f, 15000.0f, 15000.0f, 128);
    feed_sample(0.0f, 0.0f, 0.0f, 128);
    load_data_file("/mnt/c/Users/Work/Desktop/deepMIMO/RIS/renode_examples/litex-vexriscv-tensorflow-lite-demo/renode/angle.data");
    feed_sample(0.0f, 15000.0f, 15000.0f, 128);
    feed_sample(0.0f, 0.0f, 0.0f, 128);
    load_data_file("/mnt/c/Users/Work/Desktop/deepMIMO/RIS/renode_examples/litex-vexriscv-tensorflow-lite-demo/renode/circle.data");
    feed_sample(0.0f, 15000.0f, 15000.0f, 128);
    feed_sample(0.0f, 0.0f, 0.0f, 128);
    load_data_file("/mnt/c/Users/Work/Desktop/deepMIMO/RIS/renode_examples/litex-vexriscv-tensorflow-lite-demo/renode/angle.data");
    feed_sample(0.0f, 15000.0f, 15000.0f, 128);
    feed_sample(0.0f, 0.0f, 0.0f, 128);

    return kTfLiteOk;
}

bool ReadAccelerometer(tflite::ErrorReporter* error_reporter, float* input, int length) {

    //printf("ReadAccelerometer\n");
    
    // Riempie il buffer circolare come farebbe l'accelerometro reale
    // Aggiorna il buffer con nuovi dati se disponibili
    // Simula il comportamento del buffer circolare e della finestra mobile
    if (all_samples_index < all_samples.size()) {
        bufx[begin_index] = all_samples[all_samples_index][0];
        bufy[begin_index] = all_samples[all_samples_index][1];
        bufz[begin_index] = all_samples[all_samples_index][2];
        printf("begin_index 2: %d\n", begin_index);
        begin_index++;
        if (begin_index >= BUFLEN) begin_index = 0;
        printf("all_samples_index/all_samples.size(): %lu/%lu\n", all_samples_index, all_samples.size());
        all_samples_index++;
    } else {
        if (flush_buffer_counter >= 100) {
            printf("buffer scaricato\n");
            return false;
        } else {
            printf("flush_buffer_counter: %d\n", flush_buffer_counter);
            flush_buffer_counter += 1;
        }
    }
    
    if (initial && begin_index >= 100) {
        printf("Pronto per caricare buffer in input\n");
        initial = false;
    }
    
    if (initial) {
        printf("Non caricare buffer in input\n");
        return false;
    }

    int sample = 0;
    for (int i = 0; i < (length - 3); i += 3) {
        int ring_index = begin_index + sample - length / 3;
        if (ring_index < 0) {
            ring_index += BUFLEN;
        }
        input[i] = bufx[ring_index];
        input[i + 1] = bufy[ring_index];
        input[i + 2] = bufz[ring_index];
        //printf("sample: %d\n", sample);
        sample++;
    }

    return true;
}