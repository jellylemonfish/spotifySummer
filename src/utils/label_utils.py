'''
Filename: label_utils.py

Description: Save and load genres label encoder consistently.
'''

import joblib, json, numpy as np
from src.config import LABEL_ENCODER

def build_mlb_from_labels(labels_iterable):
    """Build a label encoder, with a stable class order based on data after cleanning."""
    classes = sorted(set(labels_iterable))
    mlb = MultiLabelBinarizer(classes=classes)
    mlb.fit(classes) #initialize classes
    return mlb


def save_label_encoder(encoder):
    '''Save the genres label encoder to joblib and JSON(optional).'''
    joblib.dump(encoder, LABEL_ENCODER, compress=3)
    json_path = LABEL_ENCODER.with_suffix('.json')
    if json_path:
        with open(json_path, 'w') as f:
            json.dump(encoder.classes_.tolist(), f, ensure_ascii=False, indent=2)

def load_label_encoder():
    return joblib.load(LABEL_ENCODER)
