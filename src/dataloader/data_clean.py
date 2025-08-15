"""
File Name: data_clean.py
description: Data cleaning and genre flattening for Spotify dataset. User can use it to view the genres distribution. User should edit the path before use it.
Author: Dylan Maray, Shien Zhu
Date:
"""
import pandas as pd 
import numpy as np
import ast 
from collections import Counter
from src.config import DATA_DIR, CLEAN_CSV, CLEAN_PARQUET, RAW_CSV

#Reading the CSV
#Here we request to print out all the columns in the main_dataset
main_data = pd.read_csv(RAW_CSV)
pd.set_option('display.max_columns', None)
print(main_data.head())

# The pd.read_csv() function will load "[[]]" as a string, not a Python list, so use ast.literal_eval to safely parse string 
# representations of lists:
main_data['artists_genres'] = main_data['artists_genres'].apply(ast.literal_eval)

#Defining the empty list checker
def empty_list(x): 
    return isinstance(x, list) and all(isinstance(i, list) and len(i) == 0 for i in x)

#Clean Missing Values: 
main_data['artists_genres'] = main_data['artists_genres'].apply(
    lambda x: np.nan if x == [] or empty_list(x) else x
)

#Replace empty genre fields with NaN 
cleaned_genre =  main_data['artists_genres']
print("Updated Genre:\n", cleaned_genre.head(30))


# flatten genres pre track
def flatten_genres(x):
    if isinstance(x,list):
        flat = [g for sub in x 
                if isinstance(sub,list) for g in sub]
        # normalize: keep string, strip, and lower
        flat = [str(g).strip().lower() for g in flat
                if isinstance(g,str) and g.strip() != ""]
        return sorted(set(flat))
    # keep NaN for non-list rows here
    return np.nan
# apply the flatten function
main_data['artists_genres'] = main_data['artists_genres'].apply(flatten_genres)

cleaned_genre =  main_data['artists_genres']
print("Updated Genre:\n", cleaned_genre.head(30))

#Updated Genre Distribution
# met difficult on counting, need to fix. old codes ↓
# genre_count = main_data["artists_genres"].value_counts(dropna = False)
# print("Genre Distribution:\n", genre_count )
genre_count = Counter(g for gs in main_data['artists_genres']
                      if isinstance(gs,list)
                      for g in gs)
print("\n genre counts:\n", genre_count.most_common(30))


# filter rare genres (need to discuss what is min fre,20, 30)
min_freq = 50 # adjust: 20 30 50 100
genre_keep = {g for g, c in genre_count.items() if c >= min_freq}
# print(f"\n We have {len(genre_count)} unique genres. After filtering for those asppering at least {min_freq} times, only {len(genre_keep)} remain.")
def keep_minfreq(x):
    if not isinstance(x, list):
        return np.nan
    kept = [g for g in x if g in genre_keep]
    return kept if kept else np.nan


# drop NaN, or mark unkown (need to discuss, and finish)
main_data['artists_genres'] = main_data['artists_genres'].apply(keep_minfreq)

main_data.to_csv(CLEAN_CSV, index=False)
main_data.to_parquet(CLEAN_PARQUET, index=False)