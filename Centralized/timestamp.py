import os
import pandas as pd
from datetime import timedelta

# Folder containing the CSV files
folder_path = "/home/surya/vm/darts/data"
output_folder = "/home/surya/vm/darts/data-new"

# Make sure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Define the start date as '2016-07-01 00:00:00'
start_date = pd.Timestamp('2016-07-01 00:00:00')

# Traverse through all CSV files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        # Load each CSV file
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)

        # Set the timestamp column to start from '2016-07-01 00:00:00' and increment by 5 minutes
        df['timestamp'] = pd.date_range(start=start_date, periods=len(df), freq='5T')

        # Save the updated CSV file in the output folder
        output_path = os.path.join(output_folder, filename)
        df.to_csv(output_path, index=False)

print("Timestamps updated and CSV files saved successfully!")
