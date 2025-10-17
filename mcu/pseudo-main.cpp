#include <Arduino.h>
#define BAUD_RATE 921600, CHUNK_SIZE_MAX 1024, MEAN XXX
#define INPUT_FEATURE_SIZE 1024, OUTPUT_FEATURE_SIZE 1024
// Include LiteRT for Microcontrollers headers
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h> 
#include <mlp_model_data.h> // Include model

static tflite::Model *model = nullptr;
static tflite::MicroInterpreter *interpreter = nullptr;
static TfLiteTensor *model_input = nullptr, 
static TfLiteTensor *model_output = nullptr;
static unsigned long overhead;
static float float_input[INPUT_FEATURE_SIZE];
static float input_scale_inv;
static int input_zero_point;
static constexpr int kTensorArenaSize = 32 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

void setup() {
  Serial.begin(BAUD_RATE); while (!Serial);
  int64_t ta=micros(), tb=micros(); overhead=tb - ta;

  #ifdef MODEL_IN_RAM
  uint8_t*model_ram=(uint8_t*)malloc(g_mlp_model_data_len);
  if (!model_ram)
    Serial.println("Malloc failed: not enough RAM");
    while (1) { Serial.println("STOP"); delay(1000); }
  memcpy(model_ram,g_mlp_model_data,g_mlp_model_data_len);
  model = tflite::GetModel(model_ram);
  #else
  model = tflite::GetModel(g_mlp_model_data);
  #endif

  static tflite::MicroMutableOpResolver<1> resolver;
  resolver.AddFullyConnected();

  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  interpreter->AllocateTensors();

  model_input = interpreter->input(0); 
  model_output = interpreter->output(0);

  input_scale_inv = 1.0f / model_input->params.scale;
  input_zero_point = model_input->params.zero_point;
}

void loop() {
  // Receive the input features in chunks from Serial 
  // and save them in float_input

  // Normalize and quantize input
  for (int i = 0; i < INPUT_FEATURE_SIZE; ++i)
    int32_t q = static_cast<int32_t>(
      roundf((float_input[i] - MEAN) * input_scale_inv)) 
    q += input_zero_point;
    if (q > 127) q = 127; if (q < -128) q = -128;
    model_input->data.int8[i] = static_cast<int8_t>(q);

  // Invoke interpreter to perform inference
  if (interpreter->Invoke() != kTfLiteOk) return;

  // Extract codebook index
  int64_t ta = micros();
  int max_index = 0; 
  int8_t max_value = model_output->data.int8[0];
  for (int i = 1; i < OUTPUT_FEATURE_SIZE; i++) 
    if (model_output->data.int8[i] > max_value) 
      max_index = i;
  int64_t tb = micros();
  Serial.print("extract_codebook_index_fast [#] [us]: ");
  Serial.print(max_index); Serial.print(" ");
  Serial.println(tb - ta - overhead);
}
