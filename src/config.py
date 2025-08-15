'''
Filename: config.py

Description: Dataloader configuration global settings.

Example: from src.config import DATA_DIR, RAW_CSV, MIN_FREQ, TEST_SIZE, VAL_SIZE, RANDOM_STATE
'''

from pathlib import Path

DATA_DIR = Path('data')
RAW_CSV = DATA_DIR / 'main_dataset.csv'
GENRES_TXT = DATA_DIR / 'music_genres.txt'

MIN_FREQ = 50
TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_STATE = 42

# output paths
CLEAN_CSV = DATA_DIR / f"filter_min{MIN_FREQ}.csv"
CLEAN_PARQUET = DATA_DIR / f"filter_min{MIN_FREQ}.parquet"
# VAL_CSV = DATA_DIR / f"filter_min{MIN_FREQ}_val.parquet"
# TEST_CSV = DATA_DIR / f"filter_min{MIN_FREQ}_test.parquet"
LABEL_ENCODER = DATA_DIR / 'label_encoder.joblib' #save multi-label binarizer
SPLIT_DIR = DATA_DIR / f"split_min{MIN_FREQ}"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)
