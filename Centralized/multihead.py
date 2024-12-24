import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Dropout, MultiHeadAttention, LayerNormalization, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def preprocess_data(file):
    df = pd.read_csv(file)
    df = df[["timestamp", "min_cpu", "avg_cpu", "max_cpu", "vm virtual core count", "vm memory (gb)"]].dropna()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Convert timestamp to numerical value (assuming it's in a format that can be converted to datetime)
    df['timestamp'] = pd.to_datetime(df['timestamp']).astype(int) / 10**9  # Convert to Unix timestamp
    
    data = df.values
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    return scaled_data, scaler

# Create dataset for time series
def create_dataset(data, time_step=10):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), :])
        y.append(data[i + time_step, 2])  # avg_cpu is at index 2
    return np.array(X), np.array(y)

# Define the multihead GRU attention model
def build_multihead_gru_attention_model(input_shape, num_heads=4, dropout_rate=0.2):
    inputs = Input(shape=input_shape)
    
    # GRU layer
    gru = GRU(64, return_sequences=True)(inputs)
    
    # Multi-head attention layer
    attention = MultiHeadAttention(num_heads=num_heads, key_dim=64)(gru, gru)
    
    # Add & Normalize
    attention = LayerNormalization(epsilon=1e-6)(attention + gru)
    
    # Dropout
    attention = Dropout(dropout_rate)(attention)
    
    # Global average pooling
    pooled = tf.keras.layers.GlobalAveragePooling1D()(attention)
    
    # Final dense layer
    output = Dense(1)(pooled)
    
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

# Training and evaluation function
def train_and_evaluate_model(data_folder, time_steps=10, epochs=50, batch_size=64):
    all_train_mse = []
    all_train_mae = []
    all_test_mse = []
    all_test_mae = []
    best_mse = float('inf')
    best_mae = float('inf')
    best_epoch = 0

    model = None
    for epoch in range(epochs):
        epoch_train_mse = []
        epoch_train_mae = []
        epoch_test_mse = []
        epoch_test_mae = []

        for file in os.listdir(data_folder):
            if file.endswith('.csv'):
                file_path = os.path.join(data_folder, file)
                data, scaler = preprocess_data(file_path)
                
                X, y = create_dataset(data, time_step=time_steps)

                split_index = int(len(X) * 0.8)
                X_train, X_test = X[:split_index], X[split_index:]
                y_train, y_test = y[:split_index], y[split_index:]

                # Build model on the first iteration
                if model is None:
                    model = build_multihead_gru_attention_model(input_shape=(X_train.shape[1], X_train.shape[2]))
                    callbacks = [
                        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
                    ]

                history = model.fit(
                    X_train, y_train,
                    validation_split=0.2,
                    epochs=5,  # Train for 5 epochs on each file
                    batch_size=batch_size,
                    verbose=0,
                    callbacks=callbacks
                )

                train_mse, train_mae = model.evaluate(X_train, y_train, verbose=0)
                test_mse, test_mae = model.evaluate(X_test, y_test, verbose=0)

                epoch_train_mse.append(train_mse)
                epoch_train_mae.append(train_mae)
                epoch_test_mse.append(test_mse)
                epoch_test_mae.append(test_mae)

        avg_train_mse = np.mean(epoch_train_mse)
        avg_train_mae = np.mean(epoch_train_mae)
        avg_test_mse = np.mean(epoch_test_mse)
        avg_test_mae = np.mean(epoch_test_mae)

        all_train_mse.append(avg_train_mse)
        all_train_mae.append(avg_train_mae)
        all_test_mse.append(avg_test_mse)
        all_test_mae.append(avg_test_mae)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Average Train MSE: {avg_train_mse:.4f}, MAE: {avg_train_mae:.4f}")
        print(f"Average Test MSE: {avg_test_mse:.4f}, MAE: {avg_test_mae:.4f}")

        if avg_test_mse < best_mse:
            best_mse = avg_test_mse
            best_mae = avg_test_mae
            best_epoch = epoch + 1

    print("\nBest Results:")
    print(f"Epoch: {best_epoch}")
    print(f"Test MSE: {best_mse:.4f}")
    print(f"Test MAE: {best_mae:.4f}")

    return model, all_train_mse, all_train_mae, all_test_mse, all_test_mae

# Main function to execute training
def main(data_folder):
    model, train_mse, train_mae, test_mse, test_mae = train_and_evaluate_model(data_folder)

if __name__ == "__main__":
    data_folder = "/home/surya/vm/darts/temp_data"  # Replace with your actual folder path
    main(data_folder)