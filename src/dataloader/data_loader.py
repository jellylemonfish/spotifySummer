import pandas as pd 
import numpy as np 
from collections import Counter
import ast 
import os

"""
Creating the flattened genres list 
"""

# Recursive flatten function
def flatten(l):
    for item in l:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item

# Safely parse stringified lists
def safely_parse(x):
    try:
        if isinstance(x, str):
            return ast.literal_eval(x)
        elif isinstance(x, list):
            return x
        else:
            return []
    except (ValueError, SyntaxError):
        return []

# Create flattened_genres column (first genre)
def get_first_genre(x):
    if isinstance(x, list):
        flat = list(flatten(x))
        return flat[0] if flat else np.nan
    return np.nan

""" 
Filter the dataset based on genre frequency and save the filtered version
"""

def filter_and_save_freq(df, genre_col = 'flattened_genres', min_freq = 50, output_dir = './filtered_data'):
    """
    Filter genres by minimum frequency and save to CSV and Parquet.

    Please note that the code between the ### is the way I saved my first 50 files onto my computer
    rewrite the code if necessary to ensure it runs properly on your computer
    """
    
    ###
    script_dir = os.path.dirname(__file__) #Directory of this script   
    output_dir = os.path.join(script_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)  #Ensuring the creation of the folder if it does not exist. 
    ###

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True) 

    # Here we count genre frequencies 
    genre_count = df[genre_col].value_counts()

    #Filter genres meeting minimum frequency threshold 
    valid_genres = genre_count[genre_count >= min_freq].index.tolist()
    df_filtered = df[df[genre_col].isin(valid_genres)].copy()

    # Make sure the directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Here we will prepare the filenames 
    csv_filename = os.path.join(output_dir, f"filter{min_freq}.csv")
    parquet_filename = os.path.join(output_dir, f"filter{min_freq}.parquet")

    # Debug info
    print(f"Filtered data shape: {df_filtered.shape}")
    print(f"Saving files to: {os.path.abspath(output_dir)}")

    # Save files if there's data
    if not df_filtered.empty:
        df_filtered.to_csv(csv_filename, index=False)
        df_filtered.to_parquet(parquet_filename, index=False)
        print(f"Files saved successfully:\n  - {csv_filename}\n  - {parquet_filename}")
    else:
        print("No rows to save after filtering. Consider lowering min_freq.")

    return df_filtered

if __name__ == "__main__":
    # Get path to dataset (relative to repo structure)
    script_dir = os.path.dirname(__file__)
    data_path = os.path.join(script_dir, "../spotify_data/main_dataset.csv")

    # Parse artists_genres safely
    df['artists_genres'] = df['artists_genres'].apply(safely_parse)

    # Replace empty lists with NaN
    df['artists_genres'] = df['artists_genres'].apply(lambda x: np.nan if x == [] else x)

    # Create flattened_genres column
    df['flattened_genres'] = df['artists_genres'].apply(get_first_genre)

    # Filter and save
    filtered_df = filter_and_save_freq(df, min_freq=50)
   
