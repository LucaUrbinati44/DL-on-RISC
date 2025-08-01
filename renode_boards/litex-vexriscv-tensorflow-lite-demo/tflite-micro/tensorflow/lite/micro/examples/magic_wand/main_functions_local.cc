#include "tensorflow/lite/micro/examples/magic_wand/main_functions_local.h"
#include "tensorflow/lite/micro/examples/magic_wand/accelerometer_handler_local.h"
#include "tensorflow/lite/micro/examples/magic_wand/constants.h"
#include "tensorflow/lite/micro/examples/magic_wand/gesture_predictor.h"

// include the model
#include "tensorflow/lite/micro/examples/magic_wand/magic_wand_model_data.h"
#include "tensorflow/lite/micro/examples/magic_wand/output_handler.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
//#include "tensorflow/lite/version.h"


// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;
int sample_index = 1;
int input_length;

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
    model = tflite::GetModel(g_magic_wand_model_data);
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
    static tflite::MicroMutableOpResolver<5> micro_op_resolver;  // NOLINT
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddDepthwiseConv2D();
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddMaxPool2D();
    micro_op_resolver.AddSoftmax();

    // Build/Instantiate an interpreter to run the model with.
    static tflite::MicroInterpreter static_interpreter(
        model, micro_op_resolver, tensor_arena, kTensorArenaSize); //, error_reporter);
    interpreter = &static_interpreter;


    // Tell the interpreter to allocate memory from the tensor_arena for the model's tensors.
    interpreter->AllocateTensors();

    // Obtain pointer to the model's input and output tensors. 0 represents the first (and only) input/output tensor.
    model_input = interpreter->input(0);
    model_output = interpreter->output(0);

    // Make sure the input has the properties we expect
    if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
        (model_input->dims->data[1] != 128) ||
        (model_input->dims->data[2] != kChannelNumber) ||
        (model_input->type != kTfLiteFloat32)) {
        TF_LITE_REPORT_ERROR(error_reporter, "Bad input tensor parameters in model");
        printf("%d\n", model_input->dims->size);
        printf("%d\n", model_input->dims->data[0]);
        printf("%d\n", model_input->dims->data[1]);
        printf("%d\n", model_input->type);
        return;
    }

    input_length = model_input->bytes / sizeof(float);

    TfLiteStatus setup_status = SetupAccelerometer(error_reporter);
    if (setup_status != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter, "Set up failed\n");
    } else {
        TF_LITE_REPORT_ERROR(error_reporter, "Set up succeeded\n");
    }

    printf("Exit setup\n");
}

void loop() {

    // Provide an input to the model, we set the contents of the input tensor
    // Attempt to read new data from the accelerometer.
    bool got_data = 
        ReadAccelerometer(error_reporter, model_input->data.f, input_length);
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
    int gesture_index = PredictGesture(model_output->data.f);
    printf("gesture_index: %d\n", gesture_index);

    sample_index++;

    // Produce an output
    HandleOutput(error_reporter, gesture_index);
    fflush(stdout);

}
