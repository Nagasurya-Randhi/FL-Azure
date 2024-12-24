import os
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import NBEATSModel, TCNModel, TransformerModel, TFTModel
from darts.metrics import mape, mse, mae
from darts.dataprocessing.transformers import Scaler
from darts.utils.likelihood_models import GaussianLikelihood

def process_vm_file(file_path):
    # Load data
    df = pd.read_csv(file_path)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create multivariate TimeSeries object
    ts = TimeSeries.from_dataframe(df, 'timestamp', ['min_cpu', 'max_cpu', 'avg_cpu', 'p95 max cpu'])
    
    # Scale the data
    scaler = Scaler()
    ts_scaled = scaler.fit_transform(ts)
    
    # Split data
    train, test = ts_scaled.split_before(0.8)
    
    # Define models
    models = [
        NBEATSModel(input_chunk_length=24, output_chunk_length=12, n_epochs=10),
        TCNModel(input_chunk_length=24, output_chunk_length=12, n_epochs=10),
        TransformerModel(input_chunk_length=24, output_chunk_length=12, n_epochs=10),
        TFTModel(input_chunk_length=24, output_chunk_length=12, n_epochs=10,
                 likelihood=GaussianLikelihood(),
                 add_relative_index=True,
                 add_encoders={"cyclic": {"future": ["month"]}})
    ]
    
    # Train and evaluate models
    results = {}
    for model in models:
        model.fit(train)
        forecast = model.predict(len(test))
        
        forecast = scaler.inverse_transform(forecast)
        test_original = scaler.inverse_transform(test)
        train_original = scaler.inverse_transform(train)
        
        train_forecast = model.predict(len(train))
        train_forecast = scaler.inverse_transform(train_forecast)
        
        train_mse = mse(train_original, train_forecast)
        train_mae = mae(train_original, train_forecast)
        test_mse = mse(test_original, forecast)
        test_mae = mae(test_original, forecast)
        test_mape = mape(test_original, forecast)
        
        results[type(model).__name__] = {
            'train_mse': train_mse,
            'train_mae': train_mae,
            'test_mse': test_mse,
            'test_mae': test_mae,
            'test_mape': test_mape
        }
    
    # Find best model based on test MAPE
    best_model = min(results, key=lambda x: results[x]['test_mape'])
    return best_model, results

# Iterate through files
vm_dir = '/home/surya/vm/darts/data-new'
overall_results = {}

for file_name in os.listdir(vm_dir):
    if file_name.endswith('.csv'):
        file_path = os.path.join(vm_dir, file_name)
        try:
            best_model, model_results = process_vm_file(file_path)
            overall_results[file_name] = {'best_model': best_model, 'results': model_results}
            print(f"Processed {file_name} successfully.")
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")

# Analyze overall results
for file_name, result in overall_results.items():
    print(f"\nFile: {file_name}")
    print(f"Best model: {result['best_model']}")
    print("Model results:")
    for model, metrics in result['results'].items():
        print(f"  {model}:")
        for metric, value in metrics.items():
            print(f"    {metric}: {value}")

# Calculate average metrics across all files
avg_metrics = {model: {metric: [] for metric in ['train_mse', 'train_mae', 'test_mse', 'test_mae', 'test_mape']} for model in ['NBEATSModel', 'TCNModel', 'TransformerModel', 'TFTModel']}

for result in overall_results.values():
    for model, metrics in result['results'].items():
        for metric, value in metrics.items():
            avg_metrics[model][metric].append(value)

print("\nAverage metrics across all files:")
for model, metrics in avg_metrics.items():
    print(f"{model}:")
    for metric, values in metrics.items():
        avg_value = np.mean(values)
        print(f"  Avg {metric}: {avg_value}")

# Count which model performs best most often
model_counts = {}
for result in overall_results.values():
    best_model = result['best_model']
    model_counts[best_model] = model_counts.get(best_model, 0) + 1

print("\nOverall best model counts:")
for model, count in model_counts.items():
    print(f"{model}: {count}")