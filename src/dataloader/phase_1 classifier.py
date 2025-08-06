import pandas as pd 
import numpy as np
import ast 

#Reading the CSV
#Here we request to print out all the columns in the main_dataset
main_data = pd.read_csv("/Users/dylanmaray/Desktop/6k spotify playlist/spotifySummer/spotify_data/main_dataset.csv")
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
print("Updated Genre:\n", cleaned_genre)

#Updated Genre Distribution
genre_count = main_data["artists_genres"].value_counts(dropna = False)
print("Genre Distribution:\n", genre_count )


