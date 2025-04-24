import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import os

# Assicurati che TensorFlow usi GPU (se disponibile)
print("Dispositivi disponibili:", tf.config.list_physical_devices())

# Dimensioni da testare
n_values = [100, 1000, 5000, 10000, 20000]
cpu_times = []
gpu_times = []

# Percorso dei file
os.makedirs('./', exist_ok=True)

# Esecuzione dei test
for n in n_values:
    print(f"\nEsecuzione su CPU per n = {n}")
    A = tf.random.uniform((n, n), dtype=tf.float32)
    B = tf.random.uniform((n, n), dtype=tf.float32)

    # CPU
    with tf.device('/CPU:0'):
        start = time.time()
        C_cpu = tf.matmul(A, B)
        cpu_times.append(time.time() - start)

    # GPU
    print(f"Esecuzione su GPU per n = {n}")
    with tf.device('/GPU:0'):
        A_gpu = tf.identity(A)
        B_gpu = tf.identity(B)
        start = time.time()
        C_gpu = tf.matmul(A_gpu, B_gpu)
        C_gpu_cpu = C_gpu.numpy()
        gpu_times.append(time.time() - start)

    # Verifica della correttezza
    if np.allclose(C_cpu.numpy(), C_gpu_cpu, atol=1e-4):
        print("I risultati su CPU e GPU sono uguali.")
    else:
        print("I risultati su CPU e GPU sono diversi.")

# Salvataggio dei tempi
np.save('./cpu_times.npy', np.array(cpu_times))
np.save('./gpu_times.npy', np.array(gpu_times))

# Plot dei tempi
plt.figure()
plt.semilogx(n_values, cpu_times, '-o', linewidth=2, markersize=8, label='CPU')
plt.semilogx(n_values, gpu_times, '-s', linewidth=2, markersize=8, label='GPU')
plt.xlabel('Dimensione matrice (n)')
plt.ylabel('Tempo di esecuzione (secondi)')
plt.title('Prodotto tra matrici: CPU vs GPU')
plt.legend(loc='upper left')
plt.grid(True)
plt.savefig('./TestGPU_tensorflow.png')

# Calcolo dello speedup
speedup = [c/g for c, g in zip(cpu_times, gpu_times)]
for s in speedup:
    print(f"Speedup GPU rispetto a CPU: {s:.2f}")

# Plot dello speedup
plt.figure()
plt.semilogx(n_values, speedup, '-d', color='r', linewidth=2, markersize=8)
plt.xlabel('Dimensione matrice (n)')
plt.ylabel('Speedup (CPU / GPU)')
plt.title('Speedup della GPU rispetto alla CPU')
plt.grid(True)
plt.savefig('./SpeedupGPU_tensorflow.png')