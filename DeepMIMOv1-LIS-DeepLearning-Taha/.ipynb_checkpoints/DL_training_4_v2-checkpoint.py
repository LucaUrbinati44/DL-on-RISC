import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, ReLU
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard, ProgbarLogger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import os
import scipy.io as sio

# Funzione principale per l'allenamento del modello DL
def dl_training_4(DL_input_reshaped, DL_output_reshaped, training_size, validation_size, mini_batch_size=500, max_epochs=20, use_gpu=True):
    """
    DL Beamforming Training Script
    Args:
        DL_input_reshaped (numpy.ndarray): Input reshaped dataset (features).
        DL_output_reshaped (numpy.ndarray): Output reshaped dataset (labels).
        training_size (int): Number of samples for training.
        validation_size (int): Number of samples for validation.
        mini_batch_size (int): Size of the minibatch for training.
        max_epochs (int): Maximum number of epochs for training.
        use_gpu (bool): Flag to enable or disable GPU usage.
    Returns:
        model (tensorflow.keras.Model): Trained Keras model.
        history (tensorflow.keras.callbacks.History): Training history.
    """

    # ------------------ Configurazione GPU ------------------ #
    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disabilita la GPU
    print(f"Using GPU: {tf.config.list_physical_devices('GPU')}")

    # ------------------ Preprocessing and Dataset Splitting ------------------ #
    print(f"---> DL Beamforming for Training_Size {training_size}")

    # Flatten the input and output arrays if necessary
    X = DL_input_reshaped.reshape(DL_input_reshaped.shape[0], -1).astype(np.float32)
    Y = DL_output_reshaped.reshape(DL_output_reshaped.shape[0], -1).astype(np.float32)

    # Split the dataset into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=validation_size / (training_size + validation_size), random_state=42)

    # Normalize the training data (zero-center normalization)
    # Documentation StandardScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    #  with_mean: default=True, If True, center the data before scaling. 
    #  with_std: default=True, If True, scale the data to unit variance (or equivalently, unit standard deviation).
    scaler = StandardScaler(with_std=False) # To match Matlab imageInputLayer normalization behavior
    X_train = scaler.fit_transform(X_train)
    print(f"Scaler: scale: {scaler.scale_}, mean: {scaler.mean_}, var: {scaler.var_}, n_features_in_: {scaler.n_features_in_}, n_samples_seen_: {scaler.n_samples_seen_}")
    
    # Apply the same scaling to the validation data
    X_val = scaler.transform(X_val)  # Transform the validation data using the same scaler

    print(f"Training set size: {X_train.shape}, Validation set size: {X_val.shape}")

    # ------------------ DL Model Definition ------------------ #
    # Define the neural network architecture
    model = Sequential([
        Dense(units=Y_train.shape[1], input_shape=(X_train.shape[1],), kernel_regularizer=l2(1e-4), name='Fully1'),
        ReLU(name='relu1'),
        Dropout(0.5, name='dropout1'),

        Dense(units=4 * Y_train.shape[1], kernel_regularizer=l2(1e-4), name='Fully2'),
        ReLU(name='relu2'),
        Dropout(0.5, name='dropout2'),

        Dense(units=4 * Y_train.shape[1], kernel_regularizer=l2(1e-4), name='Fully3'),
        ReLU(name='relu3'),
        Dropout(0.5, name='dropout3'),

        Dense(units=Y_train.shape[1], kernel_regularizer=l2(1e-4), name='Fully4'),
    ])

    # Compile the model with SGD optimizer and mean squared error loss
    optimizer = SGD(learning_rate=1e-1, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])

    # ------------------ Learning Rate Scheduler ------------------ #
    def lr_schedule(epoch, lr):
        if epoch > 0 and epoch % 3 == 0:
            return lr * 0.5  # Drop learning rate by factor of 0.5 every 3 epochs
        return lr

    lr_scheduler = LearningRateScheduler(lr_schedule)

    # ------------------ TensorBoard Callback ------------------ #
    tensorboard_callback = TensorBoard(log_dir="./logs", histogram_freq=1)

    # ------------------ Training Options ------------------ #
    verbose_frequency = max(1, training_size // mini_batch_size)
    progbar_logger = ProgbarLogger(count_mode='samples')

    # ------------------ DL Model Training ------------------ #
    print("Start DL training...")
    start_time = time.time()
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        batch_size=mini_batch_size,
        epochs=max_epochs,
        shuffle=True,  # Shuffle data at each epoch
        callbacks=[lr_scheduler, tensorboard_callback, progbar_logger],
        verbose=1
    )
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time / 60:.2f} minutes.")

    # ------------------ DL Model Prediction ------------------ #
    print("Start DL prediction for Figure 12...")
    Y_predicted = model.predict(X_val)
    print("Done")

    print(f"Predicted output shape: {Y_predicted.shape}")

    return model, history, Y_predicted


# ------------------ Example Usage ------------------ #
if __name__ == "__main__":

    base_folder = 'C:/Users/Work/Desktop/deepMIMO/RIS/DeepMIMOv1-LIS-DeepLearning-Taha/'
    output_folder = base_folder+'Output Python/'

    seed = 0
    np.random.seed(seed)

    simulate_data = 1

    if simulate_data == 1:
        # Simulated data for testing purposes
        dataset_size = 100
        training_size = 80 
        validation_size = 20
        M = 8
    else:
        dataset_size = 36200
        training_size = 30000
        validation_size = 6200
        M = 64*64 # 4096

    DL_input_reshaped = np.random.rand(M, 1, 1, dataset_size) 
    DL_output_reshaped = np.random.rand(1, 1, M, dataset_size)

    print(f"First sample: {DL_input_reshaped[:,0,0,0]}")

    # Call the training function
    model, history, Y_predicted = dl_training_4(DL_input_reshaped, DL_output_reshaped, training_size, validation_size)

    # Save the trained model
    model.save("trained_model.h5")
    print("Model saved as 'trained_model.h5'")

    # Save history and Y_predicted in .mat format to be imported in Matlab later
    sio.savemat(output_folder + 'history.mat', {'history': history.history})
    sio.savemat(output_folder + 'Y_predicted.mat', {'Y_predicted': Y_predicted})
    print(f"Y_predicted saved as 'Y_predicted.mat' in {output_folder}")
 
    #np.save(os.path.join(output_folder, 'history.npy'), history.history)
    #np.save(os.path.join(output_folder, 'Y_predicted.npy'), Y_predicted)
    #print("History and Y_predicted saved successfully.")