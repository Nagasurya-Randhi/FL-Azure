# Workload Prediction - CPU Trace Federated Learning

This project implements a federated learning approach for workload prediction using CPU traces.


## Dependencies

Install the required Python packages:

```
tensorflow==2.16.1
flwr==1.8.0
numpy==1.26.4
pandas==2.2.1
scikit-learn==1.4.2
pykalman==0.9.7
```

You can install these dependencies using:

```
pip install -r requirements.txt
```

## Project Structure

The main code is located in the `tft` folder.

## Uploading Code to VM

Navigate to your project directory on your local machine:

```
cd path/to/your/project
```


## Running the Project

1. First, run the file distribution script:
   ```
   python distribute_files.py
   ```

2. In one terminal, start the server:
   ```
   python start_server.py
   ```

3. In another terminal, start the clients:
   ```
   python start.py
   ```

## Results and Visualization

- The results will be saved in `metrics_history.json`.
- To visualize the results, run:
  ```
  python plotting.py
  ```

## Files

- `client.py`: Implements the federated learning client.
- `config.py`: Contains configuration settings.
- `model.py`: Defines the machine learning model.
- `requirements.txt`: Lists the project dependencies.
- `distribute_files.py`: Distributes files to clients.
- `start_server.py`: Starts the federated learning server.
- `start.py`: Initiates the federated learning process.
- `plotting.py`: Generates visualizations from the results.

## Note

Ensure all necessary permissions are set up on your Local Machine and that you have the required access to run these scripts.
