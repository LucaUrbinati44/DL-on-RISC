#include <config.h>
#include <mlp_model_data.h>

#include <dequantize_output.h>
#include <extract_codebook_index.h>
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

  unsigned long overhead;
  unsigned long overhead_esp;
  int counter = 0;

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
  constexpr int kTensorArenaSize = 114 * 1024; // con valori superiori a 114KB non va su ESP32
  uint8_t tensor_arena[kTensorArenaSize];
} // namespace

// The name of this function is important for Arduino compatibility.
void setup()
{

  // pinMode(LED_BUILTIN, OUTPUT);
  // digitalWrite(LED_BUILTIN, LOW);
  Serial.begin(115200);

  delay(10000);

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
  for (int i = 0; i < INPUT_FEATURE_SIZE; i++)
  {
    float_input_normalized[i] = (float_input[i] - mean_array[i]) / sqrtf(variance_array[i]);
  }
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

  Serial.println(codebook_index); // Scrive output sulla seriale
}