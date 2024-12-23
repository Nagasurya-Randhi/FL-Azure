import os
import json
import numpy as np
from config import DATA_DIR, NUM_CLIENTS

def distribute_files():
    all_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    np.random.shuffle(all_files)
    client_files = np.array_split(all_files, NUM_CLIENTS)
    allocation = {f"client_{i}": list(files) for i, files in enumerate(client_files)}

    with open('file_allocation.json', 'w') as f:
        json.dump(allocation, f, indent=4)

    for i, files in enumerate(client_files):
        print(f"Client {i} has {len(files)} files.")

if __name__ == "__main__":
    distribute_files()
