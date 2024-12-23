import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
import flwr as fl
import tensorflow as tf
from model import build_tft_inspired_model, create_dataset, apply_kalman_filter
from config import EPOCHS, WINDOW_SIZE, DATA_DIR, NUM_CLIENTS, SERVER_ADDRESS
import json

class TimeSeriesClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.csv_files = self.load_csv_files(client_id)
        self.epochs = EPOCHS
        
        # Initialize model shapes from first valid file
        static_shape, time_varying_shape = self._get_input_shapes()
        if static_shape is None or time_varying_shape is None:
            raise ValueError("No valid data found to initialize model")
            
        self.model = build_tft_inspired_model(static_shape, time_varying_shape)

    def _get_input_shapes(self):
        for file in self.csv_files:
            try:
                static_data, time_varying_data, _, _, _, _ = self.preprocess_data(file)
                if len(static_data) > WINDOW_SIZE:
                    X_static, X_time_varying, _ = create_dataset(static_data, time_varying_data, None, WINDOW_SIZE)
                    return X_static.shape[1:], X_time_varying.shape[1:]
            except Exception as e:
                print(f"Error processing file {file}: {str(e)}")
                continue
        return None, None

    def load_csv_files(self, client_id):
        with open('file_allocation.json', 'r') as f:
            allocation = json.load(f)
        return allocation[f"client_{client_id}"]

    def preprocess_data(self, file):
        df = pd.read_csv(file)
        df = df[["timestamp", "min_cpu", "avg_cpu", "max_cpu", "vm virtual core count", 
                "vm memory (gb)", "vm_category"]].dropna()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        
        # Convert timestamp to numerical value
        df['timestamp'] = pd.to_datetime(df['timestamp']).astype(int) / 10**9
        
        # Apply Kalman filter to time-varying features
        df['min_cpu'] = apply_kalman_filter(df['min_cpu'].values)
        df['avg_cpu'] = apply_kalman_filter(df['avg_cpu'].values)
        df['max_cpu'] = apply_kalman_filter(df['max_cpu'].values)
        
        # One-hot encode vm_category
        df = pd.get_dummies(df, columns=['vm_category'])
        
        # Ensure all vm_category columns exist (for consistent input shape)
        expected_categories = ['vm_category_t2.micro', 'vm_category_t2.small', 'vm_category_t2.medium', 
                             'vm_category_t2.large', 'vm_category_t2.xlarge', 'vm_category_t2.2xlarge']
        for cat in expected_categories:
            if cat not in df.columns:
                df[cat] = 0
        
        # Separate static and time-varying features
        static_features = ['vm virtual core count', 'vm memory (gb)'] + expected_categories
        time_varying_features = ['timestamp', 'min_cpu', 'max_cpu']
        target = ['avg_cpu']
        
        static_data = df[static_features].values
        time_varying_data = df[time_varying_features].values
        target_data = df[target].values
        
        # Scale the data
        static_scaler = MinMaxScaler(feature_range=(0, 1))
        time_varying_scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        
        static_data_scaled = static_scaler.fit_transform(static_data)
        time_varying_data_scaled = time_varying_scaler.fit_transform(time_varying_data)
        target_data_scaled = target_scaler.fit_transform(target_data)
        
        return static_data_scaled, time_varying_data_scaled, target_data_scaled, static_scaler, time_varying_scaler, target_scaler

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        total_train_mse = 0
        total_train_mae = 0
        total_samples = 0

        for file in self.csv_files:
            try:
                static_data, time_varying_data, target_data, _, _, target_scaler = self.preprocess_data(file)
                
                if len(static_data) <= WINDOW_SIZE:
                    continue
                    
                training_size = int(len(static_data) * 0.75)
                
                static_train = static_data[:training_size]
                time_varying_train = time_varying_data[:training_size]
                target_train = target_data[:training_size]
                
                X_static, X_time_varying, y = create_dataset(static_train, time_varying_train, target_train, WINDOW_SIZE)
                
                history = self.model.fit(
                    [X_static, X_time_varying], y,
                    epochs=1,
                    batch_size=64,
                    verbose=0
                )

                # Calculate metrics
                train_loss = self.model.evaluate([X_static, X_time_varying], y, verbose=0)
                mse, mae = train_loss[0], train_loss[1]
                
                total_train_mse += mse
                total_train_mae += mae
                total_samples += 1
                
            except Exception as e:
                print(f"Error processing file {file} during fitting: {str(e)}")
                continue

        if total_samples == 0:
            return self.model.get_weights(), 0, {}

        avg_train_mse = total_train_mse / total_samples
        avg_train_mae = total_train_mae / total_samples

        return self.model.get_weights(), total_samples, {
            'train_mse': float(avg_train_mse),
            'train_mae': float(avg_train_mae)
        }

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        total_mse = 0
        total_mae = 0
        total_samples = 0

        for file in self.csv_files:
            try:
                static_data, time_varying_data, target_data, _, _, target_scaler = self.preprocess_data(file)
                
                if len(static_data) <= WINDOW_SIZE:
                    continue
                    
                training_size = int(len(static_data) * 0.75)
                
                static_test = static_data[training_size:]
                time_varying_test = time_varying_data[training_size:]
                target_test = target_data[training_size:]
                
                X_static, X_time_varying, y = create_dataset(static_test, time_varying_test, target_test, WINDOW_SIZE)
                
                # Evaluate model
                test_loss = self.model.evaluate([X_static, X_time_varying], y, verbose=0)
                mse, mae = test_loss[0], test_loss[1]
                
                total_mse += mse
                total_mae += mae
                total_samples += 1
                
            except Exception as e:
                print(f"Error processing file {file} during evaluation: {str(e)}")
                continue

        if total_samples == 0:
            return float('inf'), 0, {}

        avg_mse = total_mse / total_samples
        avg_mae = total_mae / total_samples

        return float(avg_mse), total_samples, {
            'test_mse': float(avg_mse),
            'test_mae': float(avg_mae)
        }

def main():
    client_id = int(os.environ.get("CLIENT_ID", "0"))
    client = TimeSeriesClient(client_id)
    fl.client.start_numpy_client(server_address=SERVER_ADDRESS, client=client)

if __name__ == "__main__":
    main()