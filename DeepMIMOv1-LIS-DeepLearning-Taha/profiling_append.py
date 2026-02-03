import os
import subprocess

def is_windows():
    return 1 if os.name == 'nt' else 0
ISWINDOWS = is_windows()

base_folder = '/mnt/c/Users/Work/Desktop/deepMIMO/RIS/DeepMIMOv1-LIS-DeepLearning-Taha/'
if ISWINDOWS:
    base_folder = subprocess.check_output(["wsl", "wslpath", "-w", base_folder]).decode().strip()
    print(base_folder)
output_folder = os.path.join(base_folder, 'Output_Python')
pareto_plot_folder = os.path.join(base_folder, 'Pareto_plot')
mcu_profiling_folder = os.path.join(output_folder, 'Profiling_Search_MCU')

output_file_append = os.path.join(base_folder, "profiling_append.csv")

seed = 0
K_DL = 64 # subcarriers, costante (per ora)
Ur_rows = [1000, 1200]
Training_Size = [30000]
Training_Size_dd = Training_Size[0]
max_epochs = 200
active_cells = [1, 4, 8, 12, 28]
mcu_type_name_list = ['pico', 'nucleo-f446ze', 'esp32-s2-saola-tflm', 'nucleo-h753zi']
mcu_type_name_lgd_list = ['RP2040', 'STM32-F446ZE', 'ESP32-S2-SOLO', 'STM32-H753ZI']

My_ar = [32, 64]
Mz_ar = My_ar
for My, Mz in zip(My_ar, Mz_ar):
    for M_bar in active_cells:
        for mcu_type_name in mcu_type_name_list:
            end_folder = f'_seed{seed}_grid{Ur_rows[1]}_M{My}{Mz}_Mbar{M_bar}'
            end_folder_Training_Size_dd_epochs = f"{end_folder}_{Training_Size_dd}_ep{max_epochs}"
    
            output_csv = os.path.join(mcu_profiling_folder, f"profiling{end_folder_Training_Size_dd_epochs}_{mcu_type_name}.csv")

            os.system(f"cat {output_csv} >> {output_file_append}")

