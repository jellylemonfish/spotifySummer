"""
File Name: split.py

Description: Splitting the Spotify dataset into training, validation, and test sets.

Input: a parquet file named filter*.parquet (e.g. filter30.parquet, where * means the user wants to filter dataset to include only the genres more than * frequencies)

Output: three parquet files named train*.parquet, val*.parquet, and test*.parquet

"""
import numpy as np, pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifier import MultilabelStratifiedKFold
from src.config import SPLIT_DIR, CLEAN_PARQUET, TEST_SIZE, VAL_SIZE, RANDOM_STATE, MIN_FREQ
from src.utils.label_utils import build_mlb_from_labels, save_label_encoder, load_label_encoder

# load clean data

# build the genres label encoder

# build the pytorch tensor (move to training part)

#splitting


