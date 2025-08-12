import pandas as pd 
import numpy as np
from collections import Counter
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

#Clean Missing Values 
main_data['artists_genres'] = main_data['artists_genres'].apply(
    lambda x: np.nan if x == [] or empty_list(x) else x
)

#Replace empty genre fields with NaN 
cleaned_genre =  main_data['artists_genres']
print("Updated Genre:\n", cleaned_genre)

#Updated Genre Distribution
genre_count = main_data["artists_genres"].value_counts(dropna = False)
print("Genre Distribution:\n", genre_count )

#Goal: Visualize the Genre Distribution 
#Apply ast.literal_eval to safely parse string and here we drop the NaN
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
    

# Recursive flatten function
def flatten(l):
    for item in l:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item

#Safely parse each row 
genre_dropped_NaN = main_data['artists_genres'].dropna().apply(safely_parse)

# Flatten the entire list
flattened_genres = list(flatten(genre_dropped_NaN.tolist()))

#Count Frequencies in the flattened list
from collections import Counter
flattened_genre_count = Counter(flattened_genres)
top_genres = flattened_genre_count.most_common(20)

#Plotting the Top 20 genres
import matplotlib.pyplot as plt
if top_genres:
    genres, counts = zip(*top_genres)
    plt.figure(figsize=(12, 6))
    plt.bar(genres, counts)
    plt.xticks(rotation=45, ha='right')
    plt.title("Top 20 Genres")
    plt.xlabel("Genre")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show(block=False) 

else:
    print("No genres found after parsing.")


#Convert flattened genre count to data frame
flattened_genre_count_df = pd.DataFrame(flattened_genre_count.items(), columns = ["Genre", "Count"])
top_genres = flattened_genre_count_df["Genre"].head(10).tolist()

# Create a column in main_data for the first genre in each parsed list
main_data['flattened_genres'] = main_data['artists_genres'].dropna().apply(lambda x: flatten(x) if isinstance(x, list) else []).apply(lambda x: list(x)[0] if x else np.nan)

# Filter dataset
filtered_genre_df = main_data[
    main_data['flattened_genres'].isin(top_genres)
].copy()

print("Filtered dataset shape:", filtered_genre_df.shape)
print(filtered_genre_df.head())
print("Genre distribution in filtered dataset:")
print(filtered_genre_df["flattened_genres"].value_counts())

