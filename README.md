# DL-on-RISC

## TFLM commit/version:
- https://github.com/tensorflow/tflite-micro/commit/a1eb0480ed9f98e9ea5f5fb9b8e3da98ec512caf

## Toolchain

- For all MCU types:
  - -O3
  - -std=gnu++17
- For esp32:
  - framework-arduinoespressif32 @ 3.20017.241212+sha.dcc1105b 
  - tool-esptoolpy @ 2.40900.250804 (4.9.0) 
  - tool-mkfatfs @ 2.0.1 
  - tool-mklittlefs @ 1.203.210628 (2.3) 
  - tool-mkspiffs @ 2.230.0 (2.30) 
  - tool-openocd-esp32 @ 2.1100.20220706 (11.0) 
  - toolchain-riscv32-esp @ 8.4.0+2021r2-patch5 
  - toolchain-xtensa-esp32s2 @ 8.4.0+2021r2-patch5
- For stm32f4:
  - framework-arduinoststm32 @ 4.21001.250617 (2.10.1) 
  - framework-cmsis @ 2.50900.0 (5.9.0) 
  - tool-dfuutil @ 1.11.0 
  - tool-dfuutil-arduino @ 1.11.0 
  - tool-openocd @ 3.1200.0 (12.0) 
  - tool-stm32duino @ 1.0.1 
  - tool-stm32flash @ 0.7.0 
  - toolchain-gccarmnoneeabi @ 1.120301.0 (12.3.1)
- For stm32h7:
  - framework-arduinoststm32 @ 4.21001.250617 (2.10.1) 
  - framework-cmsis @ 2.50900.0 (5.9.0) 
  - tool-dfuutil @ 1.11.0 
  - tool-dfuutil-arduino @ 1.11.0 
  - tool-openocd @ 3.1200.0 (12.0) 
  - tool-stm32duino @ 1.0.1 
  - tool-stm32flash @ 0.7.0 
  - toolchain-gccarmnoneeabi @ 1.120301.0 (12.3.1)
- For pico:
  - board\_build.core = earlephilhower
  - framework-arduinopico @ 1.50201.0+sha.ded8a8a 
  - tool-mklittlefs-rp2040-earlephilhower @ 5.100300.230216 (10.3.0) 
  - tool-openocd-rp2040-earlephilhower @ 5.140200.250530 (14.2.0) 
  - tool-picotool-rp2040-earlephilhower @ 5.140200.250530 (14.2.0) 
  - tool-pioasm-rp2040-earlephilhower @ 5.140200.250530 (14.2.0) 
  - toolchain-rp2040-earlephilhower @ 5.140200.250530 (14.2.0)

