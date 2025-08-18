'''
Filename: label_utils.py

Description: Save and load genres label encoder consistently.
'''

import joblib, json, numpy as np
from pathlib import Path
from src.config import LABEL_ENCODER
from sklearn.preprocessing import MultiLabelBinarizer

def build_mlb_from_labels(labels_iterable):
    """Build a label encoder, with a stable class order based on data after cleanning."""
    #classes = sorted(set(labels_iterable))
    mlb = MultiLabelBinarizer()
    mlb.fit(labels_iterable) #initialize classes
    return mlb


def save_label_encoder(encoder, path: Path = None):
    '''Save the genres label encoder to joblib and JSON(optional).'''
    enc_path = Path(path or LABEL_ENCODER)
    enc_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(encoder, enc_path, compress=3)
    json_path = enc_path.with_suffix('.json')
    if json_path:
        with open(json_path, 'w') as f:
            json.dump(encoder.classes_.tolist(), f, ensure_ascii=False, indent=2)

    return enc_path

def load_label_encoder(path: Path = None):
    return joblib.load(path or LABEL_ENCODER)
