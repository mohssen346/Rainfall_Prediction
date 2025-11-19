# src/01_preprocess.py
import os
import pandas as pd

def merge_csv_files(input_folder: str, output_file: str) -> None:
    """
    Merge all CSV files in a folder into a single DataFrame and save it.
    """
    if not os.path.exists(input_folder):
        print(f"Error: Folder '{input_folder}' does not exist.")
        return

    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    
    if not csv_files:
        print("Warning: No CSV files found in the folder.")
        return

    dataframes = []
    for file in csv_files:
        file_path = os.path.join(input_folder, file)
        df = pd.read_csv(file_path)
        dataframes.append(df)
        print(f"Loaded: {file}")

    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df.to_csv(output_file, index=False)
    print(f"All files successfully merged into {output_file} ({len(combined_df):,} records)")

if __name__ == "__main__":
    merge_csv_files("data/raw", "data/combined_output.csv")