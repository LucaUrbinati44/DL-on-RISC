import subprocess

# Definizione dei parametri
My_ar = [32, 64]
Mz_ar = [32, 64]
My_ar = [32]
Mz_ar = [32]
#My_ar = [64]
#Mz_ar = [64]
#Training_Size = [2, 10000, 14000, 18000, 22000, 26000, 30000]
#Training_Size = [10000, 30000]
#Training_Size = [10000]
#Training_Size = [30000]
#Training_Size = [2000, 4000, 6000, 8000]
Training_Size = [2, 2000, 4000, 6000, 8000, 10000, 14000, 18000, 22000, 26000, 30000]
#Training_Size = [8000, 10000, 14000, 18000, 22000, 26000, 30000]
Training_Size = [10000, 30000]

# training
#max_epochs_load = 0
#load_model_flag = 0
#train_model_flag = 1

# loading and training
#max_epochs_load = 20
#max_epochs_load = 40 # only for My_ar = [64]
#max_epochs_load = 60 # only for My_ar = [64]
#max_epochs_load = 80 # only for My_ar = [64]
#load_model_flag = 1
#train_model_flag = 1

# loading
#max_epochs_load = 20
#max_epochs_load = 40
max_epochs_load = 60 #
#max_epochs_load = 80 # only for My_ar = [64]
#max_epochs_load = 100 # only for My_ar = [64]
load_model_flag = 1
train_model_flag = 0
predict_loaded_model_flag = 1
export_model_flag = 0
profiling_model_flag = 0

for rr in range(len(My_ar)):
    My = My_ar[rr]
    Mz = Mz_ar[rr]

    for dd in range(len(Training_Size)):
        training_size = Training_Size[dd]

        command = [
            'python', 'DL_training_4_v2_test.py',
            '--My', str(My),
            '--Mz', str(Mz),
            '--load_model_flag', str(load_model_flag),
            '--max_epochs_load', str(max_epochs_load),
            '--train_model_flag', str(train_model_flag),
            '--predict_loaded_model_flag', str(predict_loaded_model_flag),
            '--export_model_flag', str(export_model_flag),
            '--profiling_model_flag', str(profiling_model_flag),
            '--Training_Size', str(training_size)
        ]

        print(f"\n[Esecuzione] My={My}, Mz={Mz}, Training_Size={training_size}")

        # Esecuzione dello script Python come se fosse da terminale
        process = subprocess.Popen(command)

        # Attendere la fine dello script
        process.wait()

        # Check errori
        if process.returncode != 0:
            print(f"Errore durante l'esecuzione di: {command}")
            break
