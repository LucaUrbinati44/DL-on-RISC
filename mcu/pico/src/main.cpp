#include <Arduino.h>

#define COUNTER_MAX 5

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

  //int64_t ta = esp_timer_get_time();
  //int64_t tb = esp_timer_get_time();
  //overhead_esp = tb - ta;
  //Serial.print("Overhead ESP [us]: ");
  //Serial.println(overhead_esp);
}

void loop()
{
  // put your main code here, to run repeatedly:
  digitalWrite(LED_BUILTIN, HIGH);

  //int64_t ta = esp_timer_get_time();
  delay(3000);
  //int64_t tb = esp_timer_get_time();
  //Serial.print("Elapsed time ESP [us]: ");
  //Serial.println(tb - ta - overhead_esp);
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