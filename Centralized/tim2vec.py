import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, Concatenate, GlobalAveragePooling1D, Lambda
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from pykalman import KalmanFilter

def apply_kalman_filter(data):
    kf = KalmanFilter(initial_state_mean=data[0], n_dim_obs=1)
    (filtered_state_means, _) = kf.filter(data)
    return filtered_state_means

def preprocess_data(file):
    df = pd.read_csv(file)
    df = df[["timestamp", "min_cpu", "avg_cpu", "max_cpu", "vm virtual core count", "vm memory (gb)", "vm_category"]].dropna()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    df['timestamp'] = pd.to_datetime(df['timestamp']).astype(int) / 10**9
    
    df['min_cpu'] = apply_kalman_filter(df['min_cpu'].values)
    df['avg_cpu'] = apply_kalman_filter(df['avg_cpu'].values)
    df['max_cpu'] = apply_kalman_filter(df['max_cpu'].values)
    
    df = pd.get_dummies(df, columns=['vm_category'])
    
    static_features = ['vm virtual core count', 'vm memory (gb)'] + [col for col in df.columns if col.startswith('vm_category_')]
    time_varying_features = ['timestamp', 'min_cpu', 'max_cpu']
    target = ['avg_cpu']
    
    static_data = df[static_features].values
    time_varying_data = df[time_varying_features].values
    target_data = df[target].values
    
    static_scaler = MinMaxScaler(feature_range=(0, 1))
    time_varying_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    
    static_data_scaled = static_scaler.fit_transform(static_data)
    time_varying_data_scaled = time_varying_scaler.fit_transform(time_varying_data)
    target_data_scaled = target_scaler.fit_transform(target_data)
    
    return static_data_scaled, time_varying_data_scaled, target_data_scaled, static_scaler, time_varying_scaler, target_scaler

def create_dataset(static_data, time_varying_data, target_data, time_step=10):
    X_static, X_time_varying, y = [], [], []
    for i in range(len(time_varying_data) - time_step):
        X_static.append(static_data[i])
        X_time_varying.append(time_varying_data[i:(i + time_step)])
        y.append(target_data[i + time_step])
    return np.array(X_static), np.array(X_time_varying), np.array(y)

def relative_position_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates

    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])

    pos_encoding = np.concatenate([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

def transformer_block(inputs, num_heads, ff_dim, dropout=0.1):
    attention = tf.keras.layers.MultiHeadAttention(num_heads, key_dim=64)(inputs, inputs)
    x = LayerNormalization(epsilon=1e-6)(attention + inputs)
    
    ff = Dense(ff_dim, activation="relu")(x)
    ff = Dense(inputs.shape[-1])(ff)
    ff = Dropout(dropout)(ff)
    
    return LayerNormalization(epsilon=1e-6)(x + ff)

def build_improved_model(static_input_shape, time_varying_input_shape, num_heads=4, ff_dim=64, num_transformer_blocks=3, dropout_rate=0.1):
    static_inputs = Input(shape=static_input_shape)
    time_varying_inputs = Input(shape=time_varying_input_shape)
    
    static_context = Dense(64, activation='relu')(static_inputs)
    static_context = LayerNormalization(epsilon=1e-6)(static_context)
    
    # Add relative position encoding
    seq_len, feature_dim = time_varying_input_shape
    pos_encoding = Lambda(lambda x: relative_position_encoding(seq_len, feature_dim))(time_varying_inputs)
    x = time_varying_inputs + pos_encoding
    
    for _ in range(num_transformer_blocks):
        x = transformer_block(x, num_heads, ff_dim, dropout_rate)
    
    pooled = GlobalAveragePooling1D()(x)
    
    combined = Concatenate()([pooled, static_context])
    
    output = Dense(64, activation='relu')(combined)
    output = Dropout(dropout_rate)(output)
    output = Dense(32, activation='relu')(output)
    output = Dropout(dropout_rate)(output)
    output = Dense(1)(output)
    
    model = Model(inputs=[static_inputs, time_varying_inputs], outputs=output)
    
    optimizer = Adam(learning_rate=1e-3)
    
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

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
                static_data, time_varying_data, target_data, _, _, target_scaler = preprocess_data(file_path)
                
                X_static, X_time_varying, y = create_dataset(static_data, time_varying_data, target_data, time_step=time_steps)

                split_index = int(len(X_static) * 0.8)
                X_static_train, X_static_test = X_static[:split_index], X_static[split_index:]
                X_time_varying_train, X_time_varying_test = X_time_varying[:split_index], X_time_varying[split_index:]
                y_train, y_test = y[:split_index], y[split_index:]

                if model is None:
                    model = build_improved_model(
                        static_input_shape=X_static_train.shape[1:],
                        time_varying_input_shape=X_time_varying_train.shape[1:]
                    )
                    callbacks = [
                        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
                    ]

                history = model.fit(
                    [X_static_train, X_time_varying_train], y_train,
                    validation_split=0.2,
                    epochs=5,
                    batch_size=batch_size,
                    verbose=0,
                    callbacks=callbacks
                )

                train_mse, train_mae = model.evaluate([X_static_train, X_time_varying_train], y_train, verbose=0)
                test_mse, test_mae = model.evaluate([X_static_test, X_time_varying_test], y_test, verbose=0)

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

def main(data_folder):
    model, train_mse, train_mae, test_mse, test_mae = train_and_evaluate_model(data_folder)

if __name__ == "__main__":
    data_folder = "/home/surya/vm/darts/temp_data"  # Replace with your actual folder path
    main(data_folder)