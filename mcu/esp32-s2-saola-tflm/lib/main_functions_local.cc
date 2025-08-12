#include "tensorflow/lite/micro/examples/ml_on_risc/config.h"
#include "tensorflow/lite/micro/examples/ml_on_risc/dequantize_output.h"
#include "tensorflow/lite/micro/examples/ml_on_risc/extract_codebook_index.h"
#include "tensorflow/lite/micro/examples/ml_on_risc/main_functions_local.h"
#include "tensorflow/lite/micro/examples/ml_on_risc/quantize_input.h"
#include "../renode/test_set_small_10.h"
//#include "tensorflow/lite/micro/examples/ml_on_risc/read_sample_from_file_local.h"
#include "tensorflow/lite/micro/examples/ml_on_risc/read_sample_from_file_local_v2.h"
#include "tensorflow/lite/micro/examples/ml_on_risc/write_sample_to_file_local.h"
#include "tensorflow/lite/micro/examples/ml_on_risc/c_models/model_py_test_seed0_grid1200_M3232_Mbar8_10000_60_in1024_out1024_nl3_hul1024_4096_4096_model_data.h"

#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h" // outputs debug information.
#include "tensorflow/lite/micro/micro_interpreter.h" // contains code to load and run models
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h" // provides the operations used by the interpreter to run the model
#include "tensorflow/lite/schema/schema_generated.h" // contains the schema for the LiteRT FlatBuffer model file format
//#include "tensorflow/lite/version.h" // provides versioning information for the LiteRT schema


// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;
int sample_index = 1;
//int test_set_length = 1; // TODO

float input_scale;
int input_zero_point;
float output_scale;
int output_zero_point;

float float_input[INPUT_FEATURE_SIZE];
float dequantized_output[OUTPUT_FEATURE_SIZE];

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
constexpr int kTensorArenaSize = 128 * 1024; // initiale: 60kB, iniziale nuova versione tflite: 128kB.
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {

  printf("Enter setup\n");

  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  static tflite::MicroErrorReporter micro_error_reporter;  // NOLINT
  error_reporter = &micro_error_reporter; // This variable will be passed into the interpreter, which allows it to write log

  // Load a model
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model_py_test_seed0_grid1200_M3232_Mbar8_10000_60_in1024_out1024_nl3_hul1024_4096_4096_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
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
  static tflite::MicroMutableOpResolver<1> micro_op_resolver;  // NOLINT
  //micro_op_resolver.AddConv2D();
  //micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddFullyConnected();
  //micro_op_resolver.AddMaxPool2D();
  //micro_op_resolver.AddSoftmax();

  // Build/Instantiate an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize); //, error_reporter);
  interpreter = &static_interpreter;

  // Tell the interpreter to allocate memory from the tensor_arena for the model's tensors.
  interpreter->AllocateTensors();
  //if (interpreter->AllocateTensors() != kTfLiteOk) {
  //  TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
  //  return;
  //}

  // Obtain pointer to the model's input and output tensors. 0 represents the first (and only) input/output tensor.
  model_input = interpreter->input(0);
  model_output = interpreter->output(0);

  // Make sure the input has the properties we expect
  if ((model_input->dims->size != 2) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != INPUT_FEATURE_SIZE) || 
      (model_input->type != kTfLiteInt8)) {
    TF_LITE_REPORT_ERROR(error_reporter, "Bad input tensor parameters in model");
    DPRINTF("%d\n", model_input->dims->size);
    DPRINTF("%d\n", model_input->dims->data[0]);
    DPRINTF("%d\n", model_input->dims->data[1]);
    DPRINTF("%d\n", model_input->type);
    return;
  }

  if ((model_output->dims->size != 2) || (model_output->dims->data[0] != 1) ||
      (model_output->dims->data[1] != OUTPUT_FEATURE_SIZE) || 
      (model_output->type != kTfLiteInt8)) {
    TF_LITE_REPORT_ERROR(error_reporter, "Bad output tensor parameters in model");
    DPRINTF("%d\n", model_output->dims->size);
    DPRINTF("%d\n", model_output->dims->data[0]);
    DPRINTF("%d\n", model_output->dims->data[1]);
    DPRINTF("%d\n", model_output->type);
    return;
  }

  // Obtain scale and zero point
  input_scale = model_input->params.scale;
  input_zero_point = model_input->params.zero_point;
  output_scale = model_output->params.scale;
  output_zero_point = model_output->params.zero_point;

  DPRINTF("input_scale: %0.6f\n", (double)input_scale);
  DPRINTF("input_zero_point: %d\n", input_zero_point);
  DPRINTF("output_scale: %0.6f\n", (double)output_scale);
  DPRINTF("output_zero_point: %d\n\n", output_zero_point);

}

void loop() {

  //if(sample_index == test_set_length+1) {
  if(sample_index == NUM_SAMPLES+1) {
    DPRINTF("done (no more data to run)\n");
    return;
  }

  bool got_data = read_sample_from_file_local_v2(float_input);
  if (got_data) {
    //DPRINTF("got data\n");
  } else {
    DPRINTF("no data yet\n");
    return;
  }

  quantize_input(float_input, input_scale, input_zero_point, model_input->data.int8);

  //DPRINTF("Invoke for sample %d...\n", sample_index);
  TfLiteStatus invoke_status = interpreter->Invoke();
  // The possible values of TfLiteStatus, defined in common.h, are kTfLiteOk and kTfLiteError
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed on index: %d\n", sample_index++);
    return;
  }
  
  dequantize_output(model_output->data.int8, output_scale, output_zero_point, dequantized_output);

  int codebook_index = extract_codebook_index(dequantized_output);
  //DPRINTF("codebook_index %d: %d\n\n", sample_index, codebook_index);
  printf("%d\n", codebook_index);

  write_sample_to_file_local(codebook_index);
  
  sample_index++;

}
