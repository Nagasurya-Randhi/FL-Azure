import flwr as fl
import numpy as np
from config import SERVER_ADDRESS, NUM_CLIENTS

class CustomStrategy(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        num_rounds = kwargs.pop('num_rounds', 1)
        super().__init__(**kwargs)
        self.metrics_history = []
        self.best_metrics = {
            'mse': (float('inf'), -1),
            'rmse': (float('inf'), -1),
            'mae': (float('inf'), -1),
            'r2': (float('-inf'), -1)
        }
        self.num_rounds = num_rounds

    def aggregate_fit(self, rnd, results, failures):
        aggregated_weights, _ = super().aggregate_fit(rnd, results, failures)

        train_mse_values = [fit_res.metrics['train_mse'] for _, fit_res in results]
        train_rmse_values = [fit_res.metrics['train_rmse'] for _, fit_res in results]
        train_mae_values = [fit_res.metrics['train_mae'] for _, fit_res in results]

        avg_train_mse = np.mean(train_mse_values)
        avg_train_rmse = np.mean(train_rmse_values)
        avg_train_mae = np.mean(train_mae_values)

        # Print training metrics after each round
        print(f"Aggregated training metrics for round:{rnd}")
        print(f"Train MSE: {avg_train_mse:.4f}, Train RMSE: {avg_train_rmse:.4f}, Train MAE: {avg_train_mae:.4f}")
        
        return aggregated_weights, {}

    def aggregate_evaluate(self, rnd, results, failures):
        if failures:
            print(f"Round {rnd} encountered {len(failures)} failures.")

        mse_values = [r.metrics['mse'] for _, r in results]
        rmse_values = [r.metrics['rmse'] for _, r in results]
        mae_values = [r.metrics['mae'] for _, r in results]
        r2_values = [r.metrics['r2'] for _, r in results]

        avg_mse = np.mean(mse_values)
        avg_rmse = np.mean(rmse_values)
        avg_mae = np.mean(mae_values)
        avg_r2 = np.mean(r2_values)

        self.metrics_history.append((avg_mse, avg_rmse, avg_mae, avg_r2, rnd))

        if avg_mse < self.best_metrics['mse'][0]:
            self.best_metrics['mse'] = (avg_mse, rnd)
        if avg_rmse < self.best_metrics['rmse'][0]:
            self.best_metrics['rmse'] = (avg_rmse, rnd)
        if avg_mae < self.best_metrics['mae'][0]:
            self.best_metrics['mae'] = (avg_mae, rnd)
        if avg_r2 > self.best_metrics['r2'][0]:
            self.best_metrics['r2'] = (avg_r2, rnd)

        # Print metrics after each round
        print(f"Round {rnd} evaluation metrics:")
        print(f"MSE: {avg_mse:.4f}, RMSE: {avg_rmse:.4f}, MAE: {avg_mae:.4f}, R^2: {avg_r2:.4f}")

        # Print final evaluation summary after the last round
        if rnd == self.num_rounds:
            final_metrics = {
                'mse': [],
                'rmse': [],
                'mae': [],
                'r2': []
            }

            for avg_mse, avg_rmse, avg_mae, avg_r2, round_num in self.metrics_history:
                final_metrics['mse'].append((avg_mse, round_num))
                final_metrics['rmse'].append((avg_rmse, round_num))
                final_metrics['mae'].append((avg_mae, round_num))
                final_metrics['r2'].append((avg_r2, round_num))

            print("Final evaluation summary:")
            for metric, values in final_metrics.items():
                best_value, best_round = self.best_metrics[metric]
                print(f"Best {metric.upper()}: {best_value:.4f} (Round {best_round})")

        return avg_mse, {'mse': avg_mse, 'rmse': avg_rmse, 'mae': avg_mae, 'r2': avg_r2}

    # def fit_metrics_aggregation_fn(self, results):
    #     losses = [r.metrics['loss'] for _, r in results]
    #     val_losses = [r.metrics['val_loss'] for _, r in results if 'val_loss' in r.metrics]
    #     num_samples = [r.num_examples for _, r in results]
    #     total_samples = sum(num_samples)
    #     weighted_loss = sum(loss * n for loss, n in zip(losses, num_samples)) / total_samples
    #     weighted_val_loss = sum(val_loss * n for val_loss, n in zip(val_losses, num_samples)) / total_samples

    #     train_mse_values = [fit_res['train_mse'] for _, fit_res in results]
    #     train_rmse_values = [fit_res['train_rmse'] for _, fit_res in results]
    #     train_mae_values = [fit_res['train_mae'] for _, fit_res in results]

    #     avg_train_mse = np.mean(train_mse_values)
    #     avg_train_rmse = np.mean(train_rmse_values)
    #     avg_train_mae = np.mean(train_mae_values)

    #     # Print training metrics after each round
    #     print(f"Aggregated training metrics for round:")
    #     print(f"Train MSE: {avg_train_mse:.4f}, Train RMSE: {avg_train_rmse:.4f}, Train MAE: {avg_train_mae:.4f}")

    #     return {"loss": weighted_loss, "val_loss": weighted_val_loss}

def main():
    num_rounds=2
    strategy = CustomStrategy(
        fraction_fit=1.0,  # Sample all clients for training
        fraction_evaluate=1.0,  # Sample all clients for evaluation
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        on_fit_config_fn=lambda rnd: {"rnd": rnd},
        on_evaluate_config_fn=lambda rnd: {"rnd": rnd},
        # fit_metrics_aggregation_fn=CustomStrategy.fit_metrics_aggregation_fn,
        num_rounds=num_rounds  # Set number of rounds here
    )
    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        grpc_max_message_length=1024*1024*1024,
        strategy=strategy
    )

if __name__ == "__main__":
    main()
