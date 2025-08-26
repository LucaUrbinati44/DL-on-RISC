#include <config.h>
#include <mlp_model_data.h>

#include <dequantize_output.h>

#ifdef DEBUG_DEQUANTIZE
#include <extract_codebook_index.h>
#endif
#include <extract_codebook_index_fast.h>

#include <normalize_input.h>
#include <quantize_input.h>

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
  const tflite::Model *model = nullptr;
  tflite::MicroInterpreter *interpreter = nullptr;
  TfLiteTensor *model_input = nullptr;
  TfLiteTensor *model_output = nullptr;
  int sample_index = 1;
  // int test_set_length = 1; // TODO

  unsigned long overhead; // TODO: da togliere
  unsigned long overhead_esp;
  int counter = 0; // TODO: da togliere

  float input_scale;
  int input_zero_point;
  float output_scale;
  int output_zero_point;

  float float_input[INPUT_FEATURE_SIZE];
  float float_input_normalized[INPUT_FEATURE_SIZE];
  float dequantized_output[OUTPUT_FEATURE_SIZE];

  // Create an area of memory to use for input, output, and intermediate arrays.
  // The size of this will depend on the model you're using, and may need to be
  // determined by experimentation.
  constexpr int kTensorArenaSize = 32 * 1024; // con valori superiori a 114KB non va su ESP32
  uint8_t tensor_arena[kTensorArenaSize];
} // namespace

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
  // Indirizzo del modello (pesi)
  Serial.printf("Model addr: 0x%08" PRIxPTR "  region=%s  size=%d\n",
                (uintptr_t)g_mlp_model_data,
                region_of((const void *)g_mlp_model_data),
                g_mlp_model_data_len);

  // Indirizzo della tensor arena (attivazioni/buffer)
  Serial.printf("Tensor arena addr: 0x%08" PRIxPTR "  region=%s  size=%d\n",
                (uintptr_t)tensor_arena,
                region_of((const void *)tensor_arena),
                kTensorArenaSize);
}

// The name of this function is important for Arduino compatibility.
void setup()
{

  // pinMode(LED_BUILTIN, OUTPUT);
  // digitalWrite(LED_BUILTIN, LOW);
  Serial.begin(115200);

  delay(10000);

  printModelAndActivationsPlacement();

  // unsigned long t0 = micros();
  // unsigned long t1 = micros();
  // overhead = t1 - t0;
  // Serial.print("Overhead [us]: ");
  // Serial.println(overhead);

  int64_t ta = esp_timer_get_time();
  int64_t tb = esp_timer_get_time();
  overhead_esp = tb - ta;
  Serial.print("Overhead ESP [us]: ");
  Serial.println(overhead_esp);

  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  static tflite::MicroErrorReporter micro_error_reporter; // NOLINT
  error_reporter = &micro_error_reporter;                 // This variable will be passed into the interpreter, which allows it to write log

  // Load a model
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_mlp_model_data);
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

  // Obtain scale and zero point
  input_scale = model_input->params.scale;
  input_zero_point = model_input->params.zero_point;
  output_scale = model_output->params.scale;
  output_zero_point = model_output->params.zero_point;

  Serial.printf("input_scale: %0.6f\n", (double)input_scale);
  Serial.printf("input_zero_point: %d\n", input_zero_point);
  Serial.printf("output_scale: %0.6f\n", (double)output_scale);
  Serial.printf("output_zero_point: %d\n\n", output_zero_point);

  printModelAndActivationsPlacement();
}

void loop()
{
  bool got_data;

  while (1)
  {
    Serial.println("NEXT"); // Richiesta di un nuovo sample
    delay(1000);
    got_data = Serial.available(); // Attende risposta
    if (got_data)
    {
      // Serial.println("got data\n");
      break;
    }
    else
    {
      Serial.println("no data yet\n");
      delay(3000);
    }
  }

  // Legge la riga dalla seriale
  String line = Serial.readStringUntil('\n');

  // Parsing dei valori
  int index = 0;
  char *token = strtok((char *)line.c_str(), " ");
  float float_token;
  Serial.print("Ricevuto: ");
  while (token != nullptr && index < INPUT_FEATURE_SIZE)
  {
    float_token = atof(token);
    float_input[index++] = float_token;
    token = strtok(nullptr, " ");
    Serial.print(float_token, 8);
    Serial.print(" ");
  }
  Serial.println("");

  int64_t ta = esp_timer_get_time();
  normalize_input(float_input, float_input_normalized);
  int64_t tb = esp_timer_get_time();
  Serial.print("normalize_input [us]: ");
  Serial.println(tb - ta - overhead_esp);

  ta = esp_timer_get_time();
  quantize_input(float_input_normalized, input_scale, input_zero_point, model_input->data.int8);
  tb = esp_timer_get_time();
  Serial.print("quantize_input [us]: ");
  Serial.println(tb - ta - overhead_esp);

  ta = esp_timer_get_time();
  TfLiteStatus invoke_status = interpreter->Invoke();
  tb = esp_timer_get_time();
  Serial.print("interpreter_invoke [us]: ");
  Serial.println(tb - ta - overhead_esp);

  if (invoke_status != kTfLiteOk) // The possible values of TfLiteStatus, defined in common.h, are kTfLiteOk and kTfLiteError
  {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed on index: %d\n", sample_index++);
    return;
  }

#ifdef DEBUG_DEQUANTIZE
  ta = esp_timer_get_time();
  dequantize_output(model_output->data.int8, output_scale, output_zero_point, dequantized_output);
  tb = esp_timer_get_time();
  Serial.print("dequantize_output [us]: ");
  Serial.println(tb - ta - overhead_esp);

  ta = esp_timer_get_time();
  int codebook_index = extract_codebook_index(dequantized_output);
  tb = esp_timer_get_time();
  Serial.print("extract_codebook_index [#] [us]: ");
  Serial.print(codebook_index);
  Serial.print(" ");
  Serial.println(tb - ta - overhead_esp);
#endif

  ta = esp_timer_get_time();
  int codebook_index_fast = extract_codebook_index_fast(model_output->data.int8);
  tb = esp_timer_get_time();
  Serial.print("extract_codebook_index_fast [#] [us]: ");
  Serial.print(codebook_index_fast);
  Serial.print(" ");
  Serial.println(tb - ta - overhead_esp);
}