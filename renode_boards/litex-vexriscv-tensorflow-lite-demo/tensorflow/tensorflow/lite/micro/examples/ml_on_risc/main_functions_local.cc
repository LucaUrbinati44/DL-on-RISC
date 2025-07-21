/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Get started: https://ai.google.dev/edge/litert/microcontrollers/get_started

#include "tensorflow/lite/micro/examples/ml_on_risc/main_functions.h"
#include "tensorflow/lite/micro/examples/ml_on_risc/predict_codebook_index.h"
#include "tensorflow/lite/micro/examples/ml_on_risc/read_sample_from_uart_local.h"

// include the model
#include "tensorflow/lite/micro/examples/ml_on_risc/c_models/model_py_test_seed0_grid1200_M3232_Mbar8_10000_60_in1024_out1024_nl3_hul1024_4096_4096_model_data.h"

#include "tensorflow/lite/micro/micro_error_reporter.h" // outputs debug information.
#include "tensorflow/lite/micro/micro_interpreter.h" // contains code to load and run models
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h" // provides the operations used by the interpreter to run the model
#include "tensorflow/lite/schema/schema_generated.h" // contains the schema for the LiteRT FlatBuffer model file format
#include "tensorflow/lite/version.h" // provides versioning information for the LiteRT schema

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;
int sample_index = 1;
int test_set_length = 10;

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
constexpr int kTensorArenaSize = 60 * 1024;
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
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Tell the interpreter to allocate memory from the tensor_arena for the model's tensors.
  interpreter->AllocateTensors();

  // Obtain pointer to the model's input and output tensors. 0 represents the first (and only) input/output tensor.
  model_input = interpreter->input(0);
  model_output = interpreter->output(0);

  // Make sure the input has the properties we expect
  if ((model_input->dims->size != 2) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != FEATURE_SIZE) || 
      (model_input->type != kTfLiteInt8)) {
    TF_LITE_REPORT_ERROR(error_reporter, "Bad input tensor parameters in model");
    printf("%d\n", model_input->dims->size);
    printf("%d\n", model_input->dims->data[0]);
    printf("%d\n", model_input->dims->data[1]);
    printf("%d\n", model_input->type);
    return;
  }

}

void loop() {

  if(sample_index == test_set_length+1) {
    printf("done (no more data to run)\n");
    return;
  }

  // Provide an input to the model, we set the contents of the input tensor
  // Attempt to read new data from the sensor
  bool got_data = read_sample_from_uart_local(model_input->data.f);
  // If there was no new data, wait until next time.
  //if (!got_data) return;
  if (got_data) {
    printf("got data\n");
  } else {
    printf("no data yet\n");
    return;
  }

  // Run inference
  printf("Invoke for sample %d...\n", sample_index);
  TfLiteStatus invoke_status = interpreter->Invoke();
  // The possible values of TfLiteStatus, defined in common.h, are kTfLiteOk and kTfLiteError
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed on index: %d\n", sample_index++);
    return;
  }
  
  // Analyze the results to obtain a prediction. 0 represents the first (and only) output tensor
  // Obtain the output value from the tensor
  //compare_outputs(model_output->data.f); // TODO
  //printf("%d\n", model_output->dims->size);
  //printf("%d\n", model_output->dims->data[0]);
  //printf("%d\n", model_output->dims->data[1]);
  int codebook_index = predict_codebook_index(model_output->data.f);
  printf("codebook_index: %d\n", codebook_index);

  sample_index++;

  // Produce an output
  //HandleOutput(error_reporter, value);
  //fflush(stdout);

  // Check that the output value is within 0.05 of the expected value
  //TF_LITE_MICRO_EXPECT_NEAR(0., value, 0.05);

}
