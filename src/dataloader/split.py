"""
File Name: split.py

Description: Splitting the Spotify dataset into training, validation, and test sets.

Input: a parquet file named filter*.parquet (e.g. filter30.parquet, where * means the user wants to filter dataset to include only the genres more than * frequencies)
Output: three parquet files named train*.parquet, val*.parquet, and test*.parquet
Author:
Date:
"""

import pandas as pd

# read the parquet file
df = pd.read_parquet('filter30.parquet')


# build the pytorch tensor

#splitting


