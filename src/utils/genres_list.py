'''
File name: genres_list.py

Description: Load a list of music genres from music_genres.txt.
'''
from pathlib import Path

def load_genres_list(file_path: Path):
    with open(file_path, 'r', encoding='utf-8') as f:
        # Read lines and strip whitespace, one label one line
        genres = [ln.strip() for ln in f.readlines() if ln.strip()]
    return sorted(set(genres))