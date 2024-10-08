import os
import pandas as pd

# Path to the folder containing the CSV files
folder_path = r"C:\Users\alrec\Desktop\DAT Capstone\pbp_years"


# List all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Initialize an empty list to store DataFrames
dataframes = []

# Loop through each CSV file and read it into a DataFrame
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    
    # Append the DataFrame to the list
    dataframes.append(df)

# Concatenate all DataFrames into one
combined_df = pd.concat(dataframes, ignore_index=True)

# Save the combined DataFrame to a new CSV file
output_path = os.path.join(folder_path, 'raw_combined_nfl_play_by_play_2000_2023.csv')
combined_df.to_csv(output_path, index=False)

print(f"Combined CSV file saved to: {output_path}")
