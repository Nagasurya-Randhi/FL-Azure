import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Kalman Filter for outlier detection and smoothing
class KalmanFilter:
    def __init__(self, process_variance, measurement_variance):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.posteri_estimate = 0.0
        self.posteri_error_estimate = 1.0

    def update(self, measurement):
        # Prediction update
        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_variance

        # Measurement update
        blending_factor = priori_error_estimate / (priori_error_estimate + self.measurement_variance)
        self.posteri_estimate = priori_estimate + blending_factor * (measurement - priori_estimate)
        self.posteri_error_estimate = (1 - blending_factor) * priori_error_estimate

        return self.posteri_estimate

def apply_kalman_filter(data, process_variance=1e-5, measurement_variance=0.1):
    kf = KalmanFilter(process_variance, measurement_variance)
    filtered_data = []
    
    for measurement in data:
        filtered_value = kf.update(measurement)
        filtered_data.append(filtered_value)
    
    return np.array(filtered_data)

# Define Attention Mechanism
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * keys, dim=1)
        return context

# Define Gated Residual Network (GRN)
class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size):
        super(GatedResidualNetwork, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)
    
    def forward(self, x):
        return torch.sigmoid(self.linear1(x)) * x + x  # Gated residual connection

# Define Bidirectional LSTM with Attention and GRN
class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TemporalFusionTransformer, self).__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_size * 2)  # BiLSTM doubles the hidden size
        self.grn = GatedResidualNetwork(hidden_size * 2)  # GRN for feature processing
        self.fc = nn.Linear(hidden_size * 2 * 2, output_size)  # Concatenating LSTM output and context vector

    def forward(self, x):
        lstm_out, _ = self.bilstm(x)  # Shape: (batch_size, seq_len, hidden_size * 2)

        query = lstm_out[:, -1, :]  # Last time step's output as query for attention

        # Apply attention mechanism
        context = self.attention(query.unsqueeze(1), lstm_out)  # Shape: (batch_size, hidden_size * 2)

        # Apply GRN to LSTM output before concatenation
        lstm_out_processed = self.grn(lstm_out[:, -1, :])

        # Concatenate LSTM output and context vector
        combined = torch.cat((lstm_out_processed, context), dim=1)  # Shape: (batch_size, hidden_size * 4)

        out = self.fc(combined)  # Shape: (batch_size, output_size)
        
        return out

def preprocess_data(df):
    df.drop(columns=['vm_id', 'timestamp vm created', 'timestamp vm deleted'], inplace=True)

    # Convert categorical columns to numeric codes
    df['vm_category'] = pd.Categorical(df['vm_category']).codes
    
    scaler = StandardScaler()
    numerical_cols = ['min_cpu', 'max_cpu', 'avg_cpu', 'p95 max cpu', 'vm_category', 'vm virtual core count', 'vm memory (gb)']
    
    # Apply Kalman filter to each numerical column to smooth outliers
    for col in numerical_cols:
        df[col] = apply_kalman_filter(df[col].values)

    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df

def create_sequences(data, seq_length):
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length].astype(float)
        
        # Adjusted index for avg_cpu which is now the fourth column (index 3)
        target = data[i + seq_length][3].astype(float)  # Assuming avg_cpu is now the fourth column
        
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)

def train_model(model, train_loader, criterion, optimizer):
    model.train()
    for sequences, targets in train_loader:
        optimizer.zero_grad()
        
        outputs = model(sequences)
        
        loss = criterion(outputs.squeeze(), targets)
        
        loss.backward()
        
        optimizer.step()

def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for sequences, targets in test_loader:
            outputs = model(sequences)
            predictions.append(outputs.squeeze().cpu().numpy())
            actuals.append(targets.cpu().numpy())
    
    # Convert lists to numpy arrays safely
    predictions_flattened = np.concatenate(predictions) if predictions else np.array([])
    actuals_flattened = np.concatenate(actuals) if actuals else np.array([])

    return predictions_flattened, actuals_flattened

def main(data_folder, num_epochs=50):
    all_mse_train = []
    all_mae_train = []
    all_mse_test = []
    all_mae_test = []

    for file in os.listdir(data_folder):
        if file.endswith('.csv'):
            file_path = os.path.join(data_folder, file)
            df = pd.read_csv(file_path)
            df = preprocess_data(df)

            for window_size in [3, 6, 12]:
                X, y = create_sequences(df.values, window_size)

                # Split into train and test sets (70% train and 30% test)
                split_index = int(len(X) * 0.7)
                X_train, y_train = X[:split_index], y[:split_index]
                X_test, y_test = X[split_index:], y[split_index:]

                X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
                y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
                X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
                y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

                train_loader = torch.utils.data.DataLoader(list(zip(X_train_tensor, y_train_tensor)), batch_size=32,
                                                           shuffle=True)
                test_loader = torch.utils.data.DataLoader(list(zip(X_test_tensor, y_test_tensor)), batch_size=32)

                input_size = X.shape[2]
                hidden_size = 64
                output_size = 1
                
                model = TemporalFusionTransformer(input_size=input_size,
                                                   hidden_size=hidden_size,
                                                   output_size=output_size).to(device)

                criterion_train_loss_fn= nn.MSELoss()
                optimizer_train_loss_fn= optim.Adam(model.parameters(), lr=0.001)

                for epoch in range(num_epochs): 
                    train_model(model , train_loader , criterion_train_loss_fn , optimizer_train_loss_fn )

                # Evaluate on both train and test sets
                train_predictions , train_actuals= evaluate_model(model , train_loader )
                test_predictions , test_actuals= evaluate_model(model , test_loader )

                mse_train_score= mean_squared_error(train_actuals[train_actuals !=0] , train_predictions[train_actuals !=0]) if len(train_actuals[train_actuals !=0]) >0 else float('inf')
                mae_train_score= mean_absolute_error(train_actuals[train_actuals !=0] , train_predictions[train_actuals !=0]) if len(train_actuals[train_actuals !=0]) >0 else float('inf')

                mse_test_score= mean_squared_error(test_actuals[test_actuals !=0] , test_predictions[test_actuals !=0]) if len(test_actuals[test_actuals !=0]) >0 else float('inf')
                mae_test_score= mean_absolute_error(test_actuals[test_actuals !=0] , test_predictions[test_actuals !=0]) if len(test_actuals[test_actuals !=0]) >0 else float('inf')

                all_mse_train.append(mse_train_score )
                all_mae_train.append(mae_train_score )
                
                all_mse_test.append(mse_test_score )
                all_mae_test.append(mae_test_score )

                print(f"File: {file}, Window Size: {window_size}, Train MSE: {mse_train_score:.4f}, Train MAE: {mae_train_score:.4f}, Test MSE: {mse_test_score:.4f}, Test MAE: {mae_test_score:.4f}")

    avg_mse_train= np.mean(all_mse_train ) if all_mse_train else float('inf')
    avg_mae_train= np.mean(all_mae_train ) if all_mae_train else float('inf')

    avg_mse_test= np.mean(all_mse_test ) if all_mse_test else float('inf')
    avg_mae_test= np.mean(all_mae_test ) if all_mae_test else float('inf')

    print(f"Average Train MSE across all files: {avg_mse_train:.4f}")
    print(f"Average Train MAE across all files: {avg_mae_train:.4f}")
    
    print(f"Average Test MSE across all files: {avg_mse_test:.4f}")
    print(f"Average Test MAE across all files: {avg_mae_test:.4f}")

if __name__ == "__main__":
    import argparse
    
    parser= argparse.ArgumentParser(description="Train a Temporal Fusion Transformer model.")
    
    parser.add_argument("data_folder", type=str,
                        help="Path to the folder containing CSV files")
    
    parser.add_argument("--epochs", type=int,
                        default=50,
                        help="Number of epochs for training (default: 50)")
    
    args= parser.parse_args()
    
    main(args.data_folder , num_epochs=args.epochs )