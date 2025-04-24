import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, ReLU
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Funzione principale per l'allenamento del modello DL
def dl_training_4(DL_input_reshaped, DL_output_reshaped, training_size, validation_size, mini_batch_size=500, max_epochs=20):
    """
    DL Beamforming Training Script
    Args:
        DL_input_reshaped (numpy.ndarray): Input reshaped dataset (features).
        DL_output_reshaped (numpy.ndarray): Output reshaped dataset (labels).
        training_size (int): Number of samples for training.
        validation_size (int): Number of samples for validation.
        mini_batch_size (int): Size of the minibatch for training.
        max_epochs (int): Maximum number of epochs for training.
    Returns:
        model (tensorflow.keras.Model): Trained Keras model.
        history (tensorflow.keras.callbacks.History): Training history.
    """

    # ------------------ Preprocessing and Dataset Splitting ------------------ #
    print(f"---> DL Beamforming for Training_Size {training_size}")

    # Flatten the input and output arrays if necessary
    X = DL_input_reshaped.reshape(DL_input_reshaped.shape[0], -1).astype(np.float32)
    Y = DL_output_reshaped.reshape(DL_output_reshaped.shape[0], -1).astype(np.float32)

    # Normalize the input data (zero-center normalization)
    scaler = StandardScaler() #TODO
    X = scaler.fit_transform(X)

    # Split the dataset into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=validation_size / (training_size + validation_size), random_state=42)

    print(f"Training set size: {X_train.shape}, Validation set size: {X_val.shape}")

    # ------------------ DL Model Definition ------------------ #
    # Define the neural network architecture
    model = Sequential([
        Dense(units=Y_train.shape[1], input_shape=(X_train.shape[1],), name='Fully1'),
        ReLU(name='relu1'),
        Dropout(0.5, name='dropout1'),

        Dense(units=4 * Y_train.shape[1], name='Fully2'),
        ReLU(name='relu2'),
        Dropout(0.5, name='dropout2'),

        Dense(units=4 * Y_train.shape[1], name='Fully3'),
        ReLU(name='relu3'),
        Dropout(0.5, name='dropout3'),

        Dense(units=Y_train.shape[1], name='Fully4'),
    ])

    # Compile the model with SGD optimizer and mean squared error loss
    model.compile(optimizer=SGD(learning_rate=1e-1, momentum=0.9, nesterov=True),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    # ------------------ Training Options ------------------ #
    # Define training options
    validation_frequency = max(1, training_size // mini_batch_size)
    verbose_frequency = validation_frequency

    # ------------------ DL Model Training ------------------ #
    print("Start DL training...")
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        batch_size=mini_batch_size,
        epochs=max_epochs,
        verbose=1
    )
    print("Done")

    # ------------------ DL Model Prediction ------------------ #
    print("Start DL prediction for Figure 12...")
    Y_predicted = model.predict(X_val)
    print("Done")

    print(f"Predicted output shape: {Y_predicted.shape}")

    return model, history, Y_predicted


# ------------------ Example Usage ------------------ #
if __name__ == "__main__":
    # Simulated data for testing purposes
    np.random.seed(42)
    DL_input_reshaped = np.random.rand(1024, 1, 1, 10000)  # Example input data
    DL_output_reshaped = np.random.rand(1, 1, 1024, 10000)  # Example output data

    training_size = 8000
    validation_size = 2000

    # Call the training function
    model, history, Y_predicted = dl_training_4(DL_input_reshaped, DL_output_reshaped, training_size, validation_size)

    # Save the trained model
    model.save("trained_model.h5")
    print("Model saved as 'trained_model.h5'")