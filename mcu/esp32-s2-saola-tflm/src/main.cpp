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

float float_input[INPUT_FEATURE_SIZE];

void setup()
{
  Serial.begin(115200);
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
      Serial.println("no data yet");
      delay(3000);
    }
  }

  // Legge la riga dalla seriale
  String line = Serial.readStringUntil('\n');

  // Parsing dei valori
  int index = 0;
  char *token = strtok((char *)line.c_str(), " ");
  while (token != nullptr && index < INPUT_FEATURE_SIZE)
  {
    float_input[index++] = atof(token);
    token = strtok(nullptr, " ");
    Serial.println(float_input[index++])
  }

  Serial.println("Calculating...");

  delay(8000);

  Serial.println("Done");
}