"""
File Name: split.py

Description: Splitting the Spotify dataset into training, validation, and test sets.

Input: a parquet file named filter*.parquet (e.g. filter30.parquet, where * means the user wants to filter dataset to include only the genres more than * frequencies)

Output: three parquet files named train*.parquet, val*.parquet, and test*.parquet

"""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from src.config import SPLIT_DIR, CLEAN_PARQUET, TEST_SIZE, VAL_SIZE, RANDOM_STATE, MIN_FREQ, LABEL_ENCODER
from src.utils.label_utils import build_mlb_from_labels, save_label_encoder, load_label_encoder

### function
def n_splits_from_fraction(frac:float) -> int:
    "Convert a fraction to an integer number of splits."
    return max(2, int(round(1.0/float(frac))))

def first_fold_split(x, y, n:int , rs:int ):
    "Return first fold indices by multilabel stratified K fold(mskf) method. where n is the # splits(int) and rs is the random state(int)."
    mskf = MultilabelStratifiedKFold(n_splits=n, shuffle=True, random_state=rs)
    for _, test_idx in mskf.split(x, y):
        held_idx = np.asarray(test_idx)
        train_idx = np.setdiff1d(np.arange(x.shape[0]), held_idx, assume_unique=True)
        return train_idx, held_idx

    raise RuntimeError("MSKF failed to split")

### main part
def main():
    split_dir = Path(SPLIT_DIR)
    split_dir.mkdir(parents=True, exist_ok=True)

# load clean data
    df = pd.read_parquet(CLEAN_PARQUET)
    labels = df['artists_genres'].tolist()

# build the genres label encoder
    mlb = build_mlb_from_labels(labels)
    save_label_encoder(mlb, LABEL_ENCODER)

    y = mlb.transform(labels)
    x = df.index.values.reshape(-1,1)

# build the pytorch tensor (move to training part)

# splitting test
# test size = 0.2
    n_splits_here = n_splits_from_fraction(TEST_SIZE)
    temp_idx, test_idx = first_fold_split(x, y, n_splits_here, RANDOM_STATE)

# splitting validation
# validation size = 0.1, val temp = 0.1/(1-0.2)=0.125
    val_ratio = VAL_SIZE/(1-TEST_SIZE)
    n_splits_that = n_splits_from_fraction(val_ratio)
    x_temp = df.iloc[temp_idx].index.values.reshape(-1,1)
    y_temp = y[temp_idx]
    # use mskf splitting again
    train_temp, val_temp = first_fold_split(x_temp, y_temp, n_splits_that, RANDOM_STATE)

    train_idx = temp_idx[train_temp]
    val_idx = temp_idx[val_temp]

# final results
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    n =  len(df)
    print(f"Total samples: {n}, train: {len(df_train)} ({len(df_train)/n:.2%}), val: {len(df_val)} ({len(df_val)/n:.2%}), test: {len(df_test)} ({len(df_test)/n:.2%})")

# output
    tag = str(MIN_FREQ)
    df_train.to_parquet(split_dir / f"train_min{tag}.parquet", index=False)
    df_val.to_parquet(split_dir / f"val_min{tag}.parquet", index=False)
    df_test.to_parquet(split_dir / f"test_min{tag}.parquet", index=False)

if __name__ == "__main__":
    main()