import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
import flwr as fl
import tensorflow as tf
from model import create_attention_model_GRU, create_dataset, apply_kalman_filter
from config import EPOCHS, WINDOW_SIZE, UNITS, DATA_DIR, NUM_CLIENTS, SERVER_ADDRESS
import json

class TimeSeriesClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.csv_files = self.load_csv_files(client_id)
        input_shape = (WINDOW_SIZE, 1)
        self.model = create_attention_model_GRU(input_shape, UNITS)
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.epochs = EPOCHS

    def load_csv_files(self, client_id):
        with open('file_allocation.json', 'r') as f:
            allocation = json.load(f)
        return allocation[f"client_{client_id}"]

    def preprocess_data(self, file):
        df = pd.read_csv(file)
        df = df[["CPU usage [%]"]].dropna()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        data = df.values
        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(data)
        data = apply_kalman_filter(data)
        return data, scaler

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        total_train_mse = 0
        total_train_rmse = 0
        total_train_mae = 0
        total_samples = 0

        for file in self.csv_files:
            data, scaler = self.preprocess_data(file)
            if data is None or len(data) == 0:
                continue

            training_size = int(len(data) * 0.70)
            train_data, _ = data[:training_size], data[training_size:]

            if len(train_data) <= WINDOW_SIZE:
                continue

            X_train, y_train = create_dataset(train_data, WINDOW_SIZE)
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

            self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=64, verbose=0)

            # Calculate training metrics
            train_mse = self.model.evaluate(X_train, y_train, verbose=0)
            y_train_pred = self.model.predict(X_train, verbose=0)
            train_rmse = np.sqrt(train_mse)
            y_train = y_train.reshape(-1, 1)
            y_train_pred = y_train_pred.reshape(-1, 1)
            y_train = scaler.inverse_transform(y_train)
            y_train_pred = scaler.inverse_transform(y_train_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)

            total_train_mse += train_mse
            total_train_rmse += train_rmse
            total_train_mae += train_mae
            total_samples += 1

        avg_train_mse = total_train_mse / total_samples
        avg_train_rmse = total_train_rmse / total_samples
        avg_train_mae = total_train_mae / total_samples

        return self.model.get_weights(), len(X_train), {
            'train_mse': avg_train_mse,
            'train_rmse': avg_train_rmse,
            'train_mae': avg_train_mae
        }

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        total_mse = 0
        total_rmse = 0
        total_mae = 0
        total_r2 = 0
        total_samples = 0

        for file in self.csv_files:
            data, scaler = self.preprocess_data(file)
            if data is None or len(data) == 0:
                continue

            training_size = int(len(data) * 0.70)
            _, test_data = data[:training_size], data[training_size:]

            if len(test_data) <= WINDOW_SIZE:
                continue
            
            X_test, y_test = create_dataset(test_data, WINDOW_SIZE)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            if len(X_test) < 2:
                continue  # Skip this file because there are fewer than 2 samples
            
            mse = self.model.evaluate(X_test, y_test, verbose=0)
            y_pred = self.model.predict(X_test, verbose=0)
            rmse = np.sqrt(mse)
            y_test = y_test.reshape(-1, 1)
            y_pred = y_pred.reshape(-1, 1)
            y_test = scaler.inverse_transform(y_test)
            y_pred = scaler.inverse_transform(y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            total_mse += mse
            total_rmse += rmse
            total_mae += mae
            total_r2 += r2
            total_samples += 1

        avg_mse = total_mse / total_samples
        avg_rmse = total_rmse / total_samples
        avg_mae = total_mae / total_samples
        avg_r2 = total_r2 / total_samples

        return avg_mse, len(X_test), {
            'mse': avg_mse,
            'rmse': avg_rmse,
            'mae': avg_mae,
            'r2': avg_r2
        }

def main():
    client_id = int(os.environ.get("CLIENT_ID", "0"))
    client = TimeSeriesClient(client_id).to_client()
    fl.client.start_client(server_address=SERVER_ADDRESS, client=client)

if __name__ == "__main__":
    main()


