#include <Arduino.h>
// #include <TensorFlowLite_ESP32.h>

// #include "../include/config.h"
// #include "../include/model_py_test_seed0_grid1200_M3232_Mbar8_10000_60_in1024_out1024_nl3_hul1024_4096_4096_model_data.h"

// #include "../lib/dequantize_output.h"
// #include "../lib/extract_codebook_index.h"
// #include "../lib/main_functions.h"
// #include "../lib/quantize_input.h"

#include <tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h> // outputs debug information.
// #include <tensorflow/lite/micro/micro_error_reporter.h>      // outputs debug information.
#include <tensorflow/lite/micro/micro_interpreter.h>         // contains code to load and run models
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h> // provides the operations used by the interpreter to run the model
#include <tensorflow/lite/schema/schema_generated.h>         // contains the schema for the LiteRT FlatBuffer model file format
// #include "../lib/tflite-micro/tensorflow/lite/version.h"           // provides versioning information for the LiteRT schema

#define COUNTER_MAX 5

// put function declarations here:
// int myFunction(int, int);

unsigned long overhead;
unsigned long overhead_esp;
int counter = 0;

void setup()
{
  // put your setup code here, to run once:
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(115200);

  delay(10000);

  unsigned long t0 = micros();
  unsigned long t1 = micros();
  overhead = t1 - t0;
  Serial.print("Overhead [us]: ");
  Serial.println(overhead);

  int64_t ta = esp_timer_get_time();
  int64_t tb = esp_timer_get_time();
  overhead_esp = tb - ta;
  Serial.print("Overhead ESP [us]: ");
  Serial.println(overhead_esp);
}

void loop()
{
  // put your main code here, to run repeatedly:
  digitalWrite(LED_BUILTIN, HIGH);

  int64_t ta = esp_timer_get_time();
  delay(3000);
  int64_t tb = esp_timer_get_time();
  Serial.print("Elapsed time ESP [us]: ");
  Serial.println(tb - ta - overhead_esp);
  // int result = myFunction(2, 3);
  // Serial.println(result);
  digitalWrite(LED_BUILTIN, LOW);

  unsigned long t0 = micros();
  // codice da misurare
  delay(3000);
  unsigned long t1 = micros();
  Serial.print("Elapsed time [us]: ");
  Serial.println(t1 - t0 - overhead); // microsecondi

  counter++;

  if (counter == COUNTER_MAX)
  {
    Serial.println("STOP");
    counter = 0;
  }

  Serial.println("Sono il nuovo codice!");
}

// put function definitions here:
// int myFunction(int x, int y)
//{
//  return x + y;
//}