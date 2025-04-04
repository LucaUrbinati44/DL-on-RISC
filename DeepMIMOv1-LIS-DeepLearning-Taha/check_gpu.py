import tensorflow as tf
import time

# Test GPU vs CPU
def test_computation(device_name):
    # Impostare il dispositivo
    with tf.device(device_name):

        # Iniziare a misurare il tempo
        start_time = time.time()

        for i in range(10):
            # Creare due matrici casuali grandi
            mat1 = tf.random.normal([5000, 5000])
            mat2 = tf.random.normal([5000, 5000])

            # Eseguire la moltiplicazione di matrici
            result = tf.matmul(mat1, mat2)

        # Calcolare il tempo di esecuzione
        end_time = time.time()
        print(f"Tempo di esecuzione su {device_name}: {end_time - start_time} secondi")

# Esegui il test su GPU
if tf.config.list_physical_devices('GPU'):
    test_computation('/GPU:0')
else:
    print("GPU non trovata.")

# Esegui il test su CPU
test_computation('/CPU:0')
