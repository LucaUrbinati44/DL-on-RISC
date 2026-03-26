# DL-on-RISC

This is the code used to derive the results used in the following paper:

L. Urbinati, N. Decarli, F. Guidi, A. Guerra, B. M. Masini, and A. Zanella, "On-Device Deep Learning for RIS Beamforming: Design and Communication Performance," in review at IEEE Access, techRxiv preprint: DOI. Funded by EU NRRP "RESTART".

🔬Abstract

Reconfigurable Intelligent Surfaces (RISs) promise a paradigm shift in wireless communication by enabling smart propagation environments. However, traditional RIS control, which relies on base station (BS) decisions and high-volume feedback loops, struggles to meet the stringent real-time demands of next-generation systems. To overcome these limitations, this work explores direct on-board RIS control through deep learning (DL) algorithms. In particular, we shift DL-based codebook selection from the BS to the microcontroller unit (MCU) integrated into the RIS control board via a Tiny Machine Learning (TinyML) deployment pipeline. Unlike prior studies that assume such deployment feasibility without empirical validation, we conduct a hardware-aware design-space exploration that quantifies the trade-off between controller design and communication performance across a range of DL models and MCU-class
embedded devices, and identifies the resulting latency-rate Pareto frontier under realistic constraints. This paper provides design guidelines for jointly selecting RIS size, number of active elements, DL model complexity, and MCU class to meet application-specific rate and real-time requirements.

---

## Repo Author 👥

Luca Urbinati

---

```bash
📂 Repo structure
DL-on-RISC/
├── README.md                                 
├── RIS.code-workspace                        
├── environment.yml                           # Conda environment
├── code/
│   ├── MAT functions/                        # Functions (derived from Taha 2021)
│   ├── Output_Python/                        # DL training and MCU profiling results
│   ├── Pareto_plot/                          # Pareto graphs plotting scripts
│   ├── RayTracing Scenarios/                 # Ray-tracing scenarios downloaded from https://deepmimo.net
│   ├── Main_1.m                              # Main script (inspired from Taha 2021)
│   ├── UPA_codebook_generator.m              # Function (derived from Taha 2021)
│   ├── DeepMIMO_data_generator_2.m           # DeepMIMO dataset generator (inspired from Taha 2021)
│   ├── DL_data_generator_3.m                 # DeepMIMO data generation (inspired from Taha 2021)
│   ├── DL_training_4.m                       # Matlab DL training (inspired from Taha 2021)
│   ├── Fig12_plot_v2.m                       # Figure 12 plotting script for Fig. 5 of our paper (inspired from Taha 2021)
│   ├── DL_training_8_training.py             # Python script for DL training
│   ├── DL_training_4_v3_test.py              # Auxiliary Python script for DL training
│   └── serial_feeder_and_logger.py           # Python script for MCU serial communication
├── mcu/                                      # MCU firmware and deployment
│   ├── README.md                             # PlatformIO (pio) Toolchains and TFLM version
│   ├── datasheets MCUs/                      # MCUs datasheets
│   ├── esp32-s2-saola-tflm/                  # ESP32-S2-SOLO pio project
│   ├── nucleo-f446ze/                        # STM32F446ZE pio project
│   ├── nucleo-h753zi/                        # STM32H753ZI pio project
│   ├── pico/                                 # RP2040 Pico pio project
```

---

## Reference papers 📚 
- A. Taha, M. Alrabeiah, and A. Alkhateeb, "Enabling Large Intelligent Surfaces With Compressive Sensing and Deep Learning," IEEE Access, vol. 9, pp. 44304-44321, 2021. DOI: 10.1109/ACCESS.2021.3064073.
https://ieeexplore-ieee-org.ezproxy.unibo.it/document/9370097

- A. Alkhateeb, "DeepMIMO: A generic deep learning dataset for millimeter wave and massive MIMO applications," arXiv:1902.06435, 2019.
https://arxiv.org/abs/1902.06435

- "DeepMIMOv2 dataset channel generation." Accessed: Sept. 25, 2025. [Online]. 
https://deepmimo.net/versions/v2

- O. Falade, "DeepMIMO: A Generic Deep Learning Dataset for Millimeter Wave and Massive MIMO Applications to Vehicular Communications," 2023.
https://ssrn.com/abstract=4383745

- M. Singh and H. A. Kholidy, "Generic Datasets, Beamforming Vectors Prediction of 5G Cellular Networks: A Capstone Report," Dept. Network and Computer Security, SUNY Polytechnic Inst., 2020.
https://soar.suny.edu/entities/publication/e4aff4f9-883e-4031-9068-43e99d971905

---

## Reference repo 🧑‍💻

A. Taha, "LIS-DeepLearning," https://github.com/Abdelrahman-Taha/LIS-DeepLearning

---

### PlatformIO Toolchains

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

---

### TFLM commit/version:
- https://github.com/tensorflow/tflite-micro/commit/a1eb0480ed9f98e9ea5f5fb9b8e3da98ec512caf

---

## License 📄 

MIT License — Use it freely for research/academia.



