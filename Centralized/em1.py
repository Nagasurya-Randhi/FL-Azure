import os
import pandas as pd
import numpy as np
import argparse
import logging
from darts import TimeSeries
from darts.models import TFTModel
from darts.metrics import mse, mae
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_and_preprocess_data(df):
    # Check for missing values
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        logging.warning(f"Found {missing_values} missing values. Dropping rows with missing values.")
        df = df.dropna()

    # Validate numeric ranges
    numeric_columns = ['min_cpu', 'max_cpu', 'avg_cpu', 'p95 max cpu', 'vm virtual core count', 'vm memory (gb)']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].min() < 0 or df[col].max() > 100:
                logging.warning(f"{col} contains out-of-range values. Clipping to [0, 100].")
                df[col] = df[col].clip(0, 100)

    # Convert timestamp to datetime and sort
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    # Handle categorical variables
    categorical_columns = ['vm_category']
    for col in categorical_columns:
        if col in df.columns:
            df[col] = pd.Categorical(df[col]).codes

    return df

def process_file(file_path, input_chunk_length):
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Processing file: {file_path}")
        logging.info(f"File {file_path} has {len(df)} rows before processing.")

        if len(df) < input_chunk_length + 1:
            logging.warning(f"Skipping {file_path}: Not enough rows before preprocessing")
            return None

        # Validate and preprocess data
        df = validate_and_preprocess_data(df)
        
        logging.info(f"File {file_path} has {len(df)} rows after preprocessing.")

        if len(df) < input_chunk_length + 1:
            logging.warning(f"Skipping {file_path}: Not enough valid data after preprocessing")
            return None

        # Convert to TimeSeries
        target = TimeSeries.from_dataframe(df, 'timestamp', ['avg_cpu'])
        static_covariates = df[['vm virtual core count', 'vm memory (gb)']].iloc[0].to_dict()

        # Normalize static covariates
        static_scaler = StandardScaler()
        static_covariates_normalized = static_scaler.fit_transform(pd.DataFrame(static_covariates, index=[0]))
        static_covariates_normalized = dict(zip(static_covariates.keys(), static_covariates_normalized[0]))

        # Handle covariates
        covariates = TimeSeries.from_dataframe(df, 'timestamp', ['min_cpu', 'max_cpu', 'p95 max cpu'])

        scaler_target = Scaler()
        scaler_covs = Scaler()
        target = scaler_target.fit_transform(target)
        covariates = scaler_covs.fit_transform(covariates)

        # Split data
        train_target, test_target = target[:-input_chunk_length], target[-input_chunk_length:]
        train_covs, test_covs = covariates[:-input_chunk_length], covariates[-input_chunk_length:]

        # Log number of rows in the split
        logging.info(f"File {file_path}: Train target length: {len(train_target)}, Test target length: {len(test_target)}")

        if len(train_target) < input_chunk_length or len(test_target) < input_chunk_length:
            logging.warning(f"Skipping {file_path}: Not enough data after splitting")
            return None

        return train_target, test_target, train_covs, test_covs, scaler_target, static_covariates_normalized

    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}")
        return None


def create_tft_model(input_chunk_length, output_chunk_length):
    return TFTModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        hidden_size=64,
        lstm_layers=2,
        num_attention_heads=4,
        dropout=0.1,
        batch_size=32,
        n_epochs=50,
        add_relative_index=True,
        add_encoders={
            'cyclic': {'future': ['month', 'day', 'hour']},
            'datetime_attribute': {'future': ['hour', 'dayofweek', 'month']}
        },
    )

def train_and_evaluate_file(model, data):
    train_target, test_target, train_covs, test_covs, scaler_target, static_covariates_normalized = data

    # Ensure past_covariates are long enough
    if len(train_covs) < model.input_chunk_length:
        raise ValueError("Not enough historical data for the specified input_chunk_length")

    try:
        logging.info("Starting model training...")
        
        model.fit(
            series=train_target,
            past_covariates=train_covs,
            verbose=True
        )
        
        forecast = model.predict(len(test_target), past_covariates=test_covs)
        
        mse_score = mse(test_target, forecast)
        mae_score = mae(test_target, forecast)
        
        logging.info(f"Training completed. MSE: {mse_score:.4f}, MAE: {mae_score:.4f}")
        
        return mse_score, mae_score
    
    except Exception as e:
        logging.error(f"Error during model training and evaluation: {str(e)}")
        return None, None


def train_and_evaluate(data_folder, input_chunk_length=3, output_chunk_length=1):
    all_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    all_mse, all_mae = [], []
    
    for file in all_files:
        file_path = os.path.join(data_folder, file)
        
        data = process_file(file_path, input_chunk_length)
        
        if data is None:
            continue
        
        logging.info(f"Training on file: {file}")
        
        model = create_tft_model(input_chunk_length, output_chunk_length)
        
        mse_score, mae_score = train_and_evaluate_file(model, data)
        
        all_mse.append(mse_score)
        all_mae.append(mae_score)

    if all_mse and all_mae:
        avg_mse = np.mean(all_mse)
        avg_mae = np.mean(all_mae)
        
        logging.info(f"Average MSE across all files: {avg_mse:.4f}")
        logging.info(f"Average MAE across all files: {avg_mae:.4f}")
    
    else:
         logging.warning("No valid data processed. Unable to train the model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate TFT model on CPU usage data.")
    parser.add_argument("data_folder", type=str, help="Path to the folder containing CSV files")
    parser.add_argument("--input_chunk_length", type=int, default=96, help="Input chunk length for the model (default: 96)")
    parser.add_argument("--output_chunk_length", type=int, default=1, help="Output chunk length for the model (default: 1)")

    args = parser.parse_args()

    train_and_evaluate(args.data_folder, input_chunk_length=args.input_chunk_length,
                       output_chunk_length=args.output_chunk_length)
