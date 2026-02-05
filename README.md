# DL-on-RISC

This is the code used to derive the results used in the following paper:

L. Urbinati, N. Decarli, F. Guidi, A. Guerra, B. M. Masini, and A. Zanella, "On-Device Deep Learning for RIS Beamforming: Design and Communication Performance," in review at IEEE Access, techRxiv preprint: DOI. Funded by EU NRRP "RESTART".

🔬Abstract

Reconfigurable Intelligent Surfaces (RISs) promise a paradigm shift in wireless communication by enabling smart propagation environments. However, traditional RIS control, which relies on base station decisions and high-volume feedback loops, struggles to meet the stringent real-time demands of next-generation systems. To overcome these limitations, this work explores direct on-board RIS control through deep learning (DL) algorithms. Thanks to the use of tiny machine learning (TinyML), these algorithms can be executed on the microcontroller unit (MCU) integrated into the RIS control board.Unlike prior studies that assume such deployment feasibility without empirical validation, we conduct a hardware-aware analysis that quantifies the trade-off between controller design and communication performance across a range of DL models and MCU-class embedded devices, and maps the resulting latency-rate Pareto frontier under realistic constraints. This paper provides guidelines for jointly selecting RIS size, number of active cells, DL model complexity, and MCU class to meet application-specific rate and real-time requirements.

---

👥 Repo Author

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

📚 Reference papers
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

🧑‍💻 Reference repo

A. Taha, "LIS-DeepLearning," https://github.com/Abdelrahman-Taha/LIS-DeepLearning

---

📄 License

MIT License — Use it freely for research/academia.

