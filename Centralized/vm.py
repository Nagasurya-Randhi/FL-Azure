import os
import logging
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import TFTModel
from darts.metrics import mse, mae
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataProcessor:
    @staticmethod
    def load_and_prepare_data(file_path):
        try:
            df = pd.read_csv(file_path)
            logging.info(f"Columns in the file: {df.columns.tolist()}")

            if 'timestamp' not in df.columns:
                raise KeyError("Unable to identify the timestamp column")

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            columns_to_use = [col for col in df.columns if col != 'vm_id']
            df = df[columns_to_use]
            df.set_index('timestamp', inplace=True)

            df = df.sort_index()

            # Separate numeric and non-numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns

            # Resample numeric columns
            df_numeric = df[numeric_columns].resample('5min').mean()

            # Forward fill non-numeric columns
            df_non_numeric = df[non_numeric_columns].resample('5min').ffill()

            # Combine the resampled dataframes
            df = pd.concat([df_numeric, df_non_numeric], axis=1)

            # Handle categorical variables
            categorical_columns = df.select_dtypes(include=['object']).columns
            label_encoders = {}
            for col in categorical_columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le

            # Handle NaN values
            df = df.fillna(method='ffill').fillna(method='bfill')

            # Check for remaining NaN values
            if df.isnull().values.any():
                logging.warning(f"NaN values still present in the data after forward and backward fill.")
                return None, None

            return df, label_encoders
        except Exception as e:
            logging.error(f"Error loading file {file_path}: {str(e)}")
            return None, None

    @staticmethod
    def create_time_series(df):
        target = TimeSeries.from_series(df['avg_cpu'], freq='5min')
        covariates_df = df.drop('avg_cpu', axis=1)
        covariates = TimeSeries.from_dataframe(covariates_df, freq='5min')
        return target, covariates

    @staticmethod
    def split_data(target, covariates, train_ratio=0.6, val_ratio=0.2):
        total_len = len(target)
        train_len = int(total_len * train_ratio)
        val_len = int(total_len * val_ratio)

        train_target, val_target, test_target = target[:train_len], target[train_len:train_len+val_len], target[train_len+val_len:]
        train_cov, val_cov, test_cov = covariates[:train_len], covariates[train_len:train_len+val_len], covariates[train_len+val_len:]

        return (train_target, train_cov), (val_target, val_cov), (test_target, test_cov)

class ModelTrainer:
    def __init__(self, window_size):
        self.model = TFTModel(
            input_chunk_length=window_size,
            output_chunk_length=1,
            hidden_size=64,
            lstm_layers=1,
            num_attention_heads=4,
            dropout=0.1,
            batch_size=32,
            n_epochs=1,
            add_relative_index=True,
            force_reset=True
        )

    def train_and_evaluate(self, train_data, val_data, test_data):
        train_target, train_cov = train_data
        val_target, val_cov = val_data
        test_target, test_cov = test_data

        try:
            self.model.fit(train_target, past_covariates=train_cov, val_series=val_target, val_past_covariates=val_cov)

            train_predictions = self.model.predict(n=len(train_target) - self.model.input_chunk_length, series=train_target[:self.model.input_chunk_length], past_covariates=train_cov)
            test_predictions = self.model.predict(n=len(test_target) - self.model.input_chunk_length, series=test_target[:self.model.input_chunk_length], past_covariates=test_cov)

            train_actual = train_target[self.model.input_chunk_length:]
            test_actual = test_target[self.model.input_chunk_length:]

            train_mse = mse(train_actual, train_predictions)
            train_mae = mae(train_actual, train_predictions)
            test_mse = mse(test_actual, test_predictions)
            test_mae = mae(test_actual, test_predictions)

            return train_mse, train_mae, test_mse, test_mae
        except Exception as e:
            logging.error(f"Error in training or evaluation: {str(e)}")
            return None

def main(data_folder, window_sizes, epochs):
    for window_size in window_sizes:
        logging.info(f"Training with window size: {window_size}")

        trainer = ModelTrainer(window_size)

        for epoch in range(epochs):
            logging.info(f"Epoch {epoch + 1}/{epochs}")
            epoch_results = []

            for file_name in os.listdir(data_folder):
                if file_name.endswith('.csv'):
                    file_path = os.path.join(data_folder, file_name)
                    df, label_encoders = DataProcessor.load_and_prepare_data(file_path)

                    if df is None or len(df) < 3 * window_size + 3:
                        logging.warning(f"Skipping {file_name}: insufficient data (needs at least {3 * window_size + 3} rows)")
                        continue

                    target, covariates = DataProcessor.create_time_series(df)
                    train_data, val_data, test_data = DataProcessor.split_data(target, covariates)

                    if len(train_data[0]) < window_size + 1 or len(test_data[0]) < window_size + 1:
                        logging.warning(f"Skipping {file_name}: insufficient data after splitting")
                        continue

                    results = trainer.train_and_evaluate(train_data, val_data, test_data)
                    if results is not None:
                        epoch_results.append(results)

            if not epoch_results:
                logging.warning("No valid data processed in this epoch")
                continue

            avg_results = np.mean(epoch_results, axis=0)
            logging.info(f"Epoch {epoch + 1} results:")
            logging.info(f"Average Train MSE: {avg_results[0]:.4f}")
            logging.info(f"Average Train MAE: {avg_results[1]:.4f}")
            logging.info(f"Average Test MSE: {avg_results[2]:.4f}")
            logging.info(f"Average Test MAE: {avg_results[3]:.4f}")

if __name__ == "__main__":
    data_folder = "/home/surya/vm/darts/temp-new"
    window_sizes = [3, 6, 12]  # 15 minutes, 30 minutes, 1 hour
    epochs = 5  # Adjust as needed

    main(data_folder, window_sizes, epochs)
