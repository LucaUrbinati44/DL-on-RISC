#include <Arduino.h>

void setup() {
  Serial.begin(115200);
  //while (!Serial); // attende che la USB CDC sia pronta perchè enumerata dal PC
  Serial.println("Seriale pronta");
}

void loop() {
  Serial.println("Tick");
  delay(1000);
}
