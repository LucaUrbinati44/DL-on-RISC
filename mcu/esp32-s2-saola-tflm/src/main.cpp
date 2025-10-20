#include <config.h>
#include <mlp_model_data.h>

#include <dequantize_output.h>

#ifdef DEBUG_DEQUANTIZE
#include <extract_codebook_index.h>
#endif
#include <extract_codebook_index_fast.h>

#ifdef DEBUG_QUANTIZE
#include <normalize_input.h>
#include <quantize_input.h>
#endif

#include <tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h> // outputs debug information.
// #include <tensorflow/lite/micro/micro_error_reporter.h>      // outputs debug information.
#include <tensorflow/lite/micro/micro_interpreter.h>         // contains code to load and run models
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h> // provides the operations used by the interpreter to run the model
#include <tensorflow/lite/schema/schema_generated.h>         // contains the schema for the LiteRT FlatBuffer model file format
// #include "../lib/tflite-micro/tensorflow/lite/version.h"           // provides versioning information for the LiteRT schema

// Globals, used for compatibility with Arduino-style sketches.
namespace
{
  tflite::ErrorReporter *error_reporter = nullptr;

#ifdef MODEL_IN_RAM
  uint8_t *model_ram = nullptr;
#endif
  const tflite::Model *model = nullptr;

  tflite::MicroInterpreter *interpreter = nullptr;
  TfLiteTensor *model_input = nullptr;
  TfLiteTensor *model_output = nullptr;

  // unsigned long overhead; // TODO: da togliere
  unsigned long overhead;

  float input_scale;
  int input_zero_point;
  float output_scale;
  int output_zero_point;
  float input_scale_inv;

  //float chunk_buf[CHUNK_SIZE_MAX];
  uint8_t chunk_buf[CHUNK_SIZE_MAX * sizeof(float)];
  float float_input[INPUT_FEATURE_SIZE];
  float float_input_normalized[INPUT_FEATURE_SIZE];
  float dequantized_output[OUTPUT_FEATURE_SIZE];

  // Create an area of memory to use for input, output, and intermediate arrays.
  // The size of this will depend on the model you're using, and may need to be
  // determined by experimentation.
  constexpr int kTensorArenaSize = 32 * 1024; // con valori superiori a 114KB non va su ESP32, confronti fatti con 114KB
  uint8_t tensor_arena[kTensorArenaSize];
} // namespace
// Con namespace { ... } (anonymous namespace). Tutte le variabili e funzioni dichiarate dentro hanno linkage interno → sono visibili solo all’interno di quel file .cpp.
// È l’equivalente C++ di usare static su variabili globali in C.
// Senza namespace { ... } (variabili globali pure), Le variabili diventano globali con linkage esterno.
// Se il codice è solo questo file .ino/.cpp → puoi rimuovere tranquillamente il namespace {} e risparmiare 2 righe senza effetti collaterali.
// Se pensi di integrare in un progetto più grande (più file .cpp) → meglio lasciare il namespace {} per evitare conflitti futuri.



/*
static inline bool in_range(uintptr_t a, uintptr_t start, uintptr_t len)
{
  return (a >= start) && (a < (start + len));
}

// Segmenti presi dal tuo .map
static const uintptr_t IRAM0_0_START = 0x40024000UL;
static const uintptr_t IRAM0_0_LEN = 0x0002C000UL;

static const uintptr_t IRAM0_2_START = 0x40080020UL;
static const uintptr_t IRAM0_2_LEN = 0x0077FFE0UL;

static const uintptr_t DRAM0_0_START = 0x3FFB4000UL;
static const uintptr_t DRAM0_0_LEN = 0x0002C000UL;

static const uintptr_t DROM0_0_START = 0x3F000020UL;
static const uintptr_t DROM0_0_LEN = 0x003EFFE0UL;

static const uintptr_t RTC_IRAM_START = 0x40070000UL;
static const uintptr_t RTC_IRAM_LEN = 0x00001FF0UL;

static const uintptr_t RTC_SLOW_START = 0x50000000UL;
static const uintptr_t RTC_SLOW_LEN = 0x00002000UL;

static const uintptr_t RTC_DATA_START = 0x3FF9E000UL;
static const uintptr_t RTC_DATA_LEN = 0x00001FF0UL;

static const uintptr_t EXTRAM_START = 0x3F500000UL; // PSRAM, se presente
static const uintptr_t EXTRAM_LEN = 0x00A80000UL;

static const char *region_of(const void *p)
{
  uintptr_t a = (uintptr_t)p;

  if (in_range(a, DRAM0_0_START, DRAM0_0_LEN))
    return "DRAM";
  if (in_range(a, DROM0_0_START, DROM0_0_LEN))
    return "DROM (flash-mapped const)";
  if (in_range(a, IRAM0_0_START, IRAM0_0_LEN))
    return "IRAM (exec)";
  if (in_range(a, IRAM0_2_START, IRAM0_2_LEN))
    return "IRAM/IROM (exec/XIP)";
  if (in_range(a, RTC_IRAM_START, RTC_IRAM_LEN))
    return "RTC_IRAM";
  if (in_range(a, RTC_SLOW_START, RTC_SLOW_LEN))
    return "RTC_SLOW";
  if (in_range(a, RTC_DATA_START, RTC_DATA_LEN))
    return "RTC_DATA";
  if (in_range(a, EXTRAM_START, EXTRAM_LEN))
    return "EXTRAM (PSRAM)";

  return "UNKNOWN";
}

void printModelAndActivationsPlacement()
{
  // Indirizzo del modello (pesi) in Flash
  Serial.printf("Model addr: 0x%08" PRIxPTR "  region=%s  size=%d\n",
                (uintptr_t)g_mlp_model_data,
                region_of((const void *)g_mlp_model_data),
                g_mlp_model_data_len);

  // Indirizzo del modello (pesi) in RAM
  Serial.printf("Model addr: 0x%08" PRIxPTR "  region=%s  size=%d\n",
                (uintptr_t)model_ram,
                region_of((const void *)model_ram),
                g_mlp_model_data_len);

  // Indirizzo della tensor arena (attivazioni/buffer)
  Serial.printf("Tensor arena addr: 0x%08" PRIxPTR "  region=%s  size=%d\n",
                (uintptr_t)tensor_arena,
                region_of((const void *)tensor_arena),
                kTensorArenaSize);
}
*/

// The name of this function is important for Arduino compatibility.
void setup()
{

  // pinMode(LED_BUILTIN, OUTPUT);
  // digitalWrite(LED_BUILTIN, LOW);
  Serial.begin(BAUD_RATE);
  Serial.setTimeout(10000);
  //delay(3000);
  while (!Serial); // Wait until someone opens the serial communication

  // unsigned long t0 = micros();
  // unsigned long t1 = micros();
  // overhead = t1 - t0;
  // Serial.print("Overhead [us]: ");
  // Serial.println(overhead);

  int64_t ta = micros();
  int64_t tb = micros();
  overhead = tb - ta;
  Serial.print("Overhead [us]: ");
  Serial.println(overhead);

  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  static tflite::MicroErrorReporter micro_error_reporter; // NOLINT
  error_reporter = &micro_error_reporter;                 // This variable will be passed into the interpreter, which allows it to write log

// Load a model
#ifdef MODEL_IN_RAM
  // Copy move from Flash to RAM
  Serial.println("*** Copy move from Flash to RAM");
  model_ram = (uint8_t *)malloc(g_mlp_model_data_len);
  if (!model_ram) {
    Serial.println("ERRORE: malloc fallita, memoria insufficiente!");
    while (1) {
      Serial.println("STOP");
      delay(1000);
    }
  }
  memcpy(model_ram, g_mlp_model_data, g_mlp_model_data_len);
  model = tflite::GetModel(model_ram);
#else
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_mlp_model_data);
#endif

  if (model->version() != TFLITE_SCHEMA_VERSION)
  {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // The MicroMutableOpResolver will be used by the interpreter to register and access the operations that are used by the model.
  // It requires a template parameter indicating the number of ops that will be registered (i.e., the number in <>).
  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  static tflite::MicroMutableOpResolver<1> micro_op_resolver; // NOLINT
  // micro_op_resolver.AddConv2D();
  // micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddFullyConnected();
  // micro_op_resolver.AddMaxPool2D();
  // micro_op_resolver.AddSoftmax();

  // Build/Instantiate an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize); //, error_reporter);
  interpreter = &static_interpreter;

  // Tell the interpreter to allocate memory from the tensor_arena for the model's tensors.
  interpreter->AllocateTensors();
  // if (interpreter->AllocateTensors() != kTfLiteOk) {
  //   TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
  //   return;
  // }

  // Obtain pointer to the model's input and output tensors. 0 represents the first (and only) input/output tensor.
  model_input = interpreter->input(0);
  model_output = interpreter->output(0);

  // Make sure the input has the properties we expect
  if ((model_input->dims->size != 2) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != INPUT_FEATURE_SIZE) ||
      (model_input->type != kTfLiteInt8))
  {
    TF_LITE_REPORT_ERROR(error_reporter, "Bad input tensor parameters in model");
    Serial.println(model_input->dims->size);
    Serial.println(model_input->dims->data[0]);
    Serial.println(model_input->dims->data[1]);
    Serial.println(model_input->type);
    return;
  }

  if ((model_output->dims->size != 2) || (model_output->dims->data[0] != 1) ||
      (model_output->dims->data[1] != OUTPUT_FEATURE_SIZE) ||
      (model_output->type != kTfLiteInt8))
  {
    TF_LITE_REPORT_ERROR(error_reporter, "Bad output tensor parameters in model");
    Serial.println(model_output->dims->size);
    Serial.println(model_output->dims->data[0]);
    Serial.println(model_output->dims->data[1]);
    Serial.println(model_output->type);
    return;
  }

  //Serial.print("mean_array[0]: ");
  //Serial.println((double)mean_array[0], 22);
  Serial.print("mean_array: ");
  Serial.println((double)mean_array, 22);
  //Serial.print("variance_array_sqrt_inv[0]: ");
  //Serial.println((double)variance_array_sqrt_inv[0], 22);

  // Obtain scale and zero point
  input_scale = model_input->params.scale;
  input_zero_point = model_input->params.zero_point;
  output_scale = model_output->params.scale;
  output_zero_point = model_output->params.zero_point;

  Serial.print("input_scale: ");
  Serial.println((double)input_scale, 22);
  Serial.print("input_zero_point: ");
  Serial.println((double)input_zero_point, 22);
  Serial.print("output_scale: ");
  Serial.println((double)output_scale, 22);
  Serial.print("output_zero_point: ");
  Serial.println((double)output_zero_point, 22);

  input_scale_inv = 1.0f / input_scale;

  // printModelAndActivationsPlacement();
  
  Serial.print("CPU Frequency: ");
  Serial.print(getCpuFrequencyMhz());
  Serial.println(" MHz");
}

// ------------------------------------------------------------

void loop()
{

  // --- RICEZIONE CHUNK ---
  uint32_t features_received = 0;
  while (features_received < INPUT_FEATURE_SIZE) {
    //uint16_t chunk_size = min((uint32_t)CHUNK_SIZE_MAX, total_features - features_received);
    uint16_t chunk_size = min((uint32_t)CHUNK_SIZE_MAX, INPUT_FEATURE_SIZE - features_received);
    uint32_t chunk_size_in_bytes = chunk_size * sizeof(float);

    // Legge dati dalla porta seriale e li memorizza in un buffer.
    // Il primo argomento (p + bytes_received) è il puntatore alla posizione corrente nel buffer dove scrivere i nuovi dati.
    // Il secondo argomento (chunk_size_in_bytes - bytes_received) è il numero di byte da leggere (quanti ne mancano per completare il chunk).
    uint32_t bytes_received = 0;
    while (bytes_received < chunk_size_in_bytes) {
      while (Serial.available()) Serial.read(); // azzera il buffer seriale da residui
      int r = Serial.readBytes(chunk_buf + bytes_received, chunk_size_in_bytes - bytes_received);
      if (r <= 0) { 
        //Serial.println("ERR timeout or no data yet");
        Serial.println("Sono ESP32");
        //Serial.print("Overhead ESP [us]: ");
        //Serial.println(overhead);
        ////Serial.print("mean_array[0]: ");
        ////Serial.println((double)mean_array[0], 22);
        //Serial.print("mean_array: ");
        //Serial.println((double)mean_array, 22);
        ////Serial.print("variance_array_sqrt_inv[0]: ");
        ////Serial.println((double)variance_array_sqrt_inv[0], 22);
        //Serial.print("input_scale: ");
        //Serial.println((double)input_scale, 22);
        //Serial.print("input_zero_point: ");
        //Serial.println((double)input_zero_point, 22);
        ////Serial.print("output_scale: ");
        ////Serial.println((double)output_scale, 22);
        ////Serial.print("output_zero_point: ");
        ////Serial.println((double)output_zero_point, 22);
        //Serial.print("CPU Frequency: ");
        //Serial.print(getCpuFrequencyMhz());
        //Serial.println(" MHz");
        Serial.println("NEXT");
        Serial.flush();
        delay(1000);
        return;
      }
      Serial.println("ACK");
      bytes_received += r;
    }
    
    // Ora chunk_buf contiene `chunk_size` float (little-endian)
    // Copia nell'input tensor
    float *chunk_as_float = reinterpret_cast<float*>(chunk_buf);
    for (int i = 0; i < chunk_size; i++) {
      float_input[features_received + i] = chunk_as_float[i];
    }

    features_received += chunk_size;
  
    //Serial.print("Received ");
    //Serial.print(features_received + chunk_size);
    //Serial.print("/");
    //Serial.print(total_features);
    //Serial.print(" features (");
    //Serial.print((features_received+chunk_size)/total_features*100);
    //Serial.println(")");
  }
  
  //if (features_received == INPUT_FEATURE_SIZE) {
  //  Serial.println("All features received");
  //} else {
  //  Serial.println("ERROR: unexpected total number of received features");
  //  return;
  //}
  if (features_received != INPUT_FEATURE_SIZE) {
    Serial.println("ERROR: unexpected total number of received features");
    return;
  }

  // ------------------------------------------------------------------------

#ifdef DEBUG_QUANTIZE
  //int64_t ta = esp_timer_get_time();
  int64_t ta = micros();
  normalize_input(float_input, float_input_normalized);
  int64_t tb = micros();
  Serial.print("normalize_input [us]: ");
  Serial.println(tb - ta - overhead);

  ta = micros();
  // quantize_input(float_input_normalized, input_scale, input_zero_point, model_input->data.int8);
  quantize_input(float_input_normalized, input_scale_inv, input_zero_point, model_input->data.int8);
  tb = micros();
  Serial.print("quantize_input [us]: ");
  Serial.println(tb - ta - overhead);
#endif

#ifdef DEBUG_QUANTIZE
  ta = micros();
#else
  int64_t ta = micros();
#endif
  #ifdef ENABLE_UNROLL_NORMALIZE
  #pragma GCC unroll 1024
  #endif
  for (int i = 0; i < INPUT_FEATURE_SIZE; ++i)
  {    
    int32_t q = static_cast<int32_t>(roundf((float_input[i] - mean_array) * input_scale_inv)) + input_zero_point;
    if (q > 127)
      q = 127;
    if (q < -128)
      q = -128;
    model_input->data.int8[i] = static_cast<int8_t>(q);
  }
#ifdef DEBUG_QUANTIZE
  tb = micros();
#else
  int64_t tb = micros();
#endif
  Serial.print("normalize_and_quantize_input [us]: ");
  Serial.println(tb - ta - overhead);

  ta = micros();
  TfLiteStatus invoke_status = interpreter->Invoke();
  tb = micros();
  Serial.print("interpreter_invoke [us]: ");
  Serial.println(tb - ta - overhead);

  if (invoke_status != kTfLiteOk) // The possible values of TfLiteStatus, defined in common.h, are kTfLiteOk and kTfLiteError
  {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed\n");
    return;
  }

#ifdef DEBUG_DEQUANTIZE
  ta = micros();
  dequantize_output(model_output->data.int8, output_scale, output_zero_point, dequantized_output);
  tb = micros();
  Serial.print("dequantize_output [us]: ");
  Serial.println(tb - ta - overhead);

  ta = micros();
  int codebook_index = extract_codebook_index(dequantized_output);
  tb = micros();
  Serial.print("extract_codebook_index [#] [us]: ");
  Serial.print(codebook_index);
  Serial.print(" ");
  Serial.println(tb - ta - overhead);
#endif

  ta = micros();
  int codebook_index_fast = extract_codebook_index_fast(model_output->data.int8);
  tb = micros();
  Serial.print("extract_codebook_index_fast [#] [us]: ");
  Serial.print(codebook_index_fast);
  Serial.print(" ");
  Serial.println(tb - ta - overhead);

}
