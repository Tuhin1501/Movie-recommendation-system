# Movie-recommendation-system
Built a recommendation system that recommends movies based on user input using cosine similarity filter.

#code:
import pandas as pd
import numpy as np
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
Movies = pd.read_csv(r"C:\Users\t1u5h\Downloads\movies.csv")
Movies.shape
Movies.columns
Specifications = ['title','genres','keywords','tagline','cast','director']
print (Specifications)
for specs in Specifications:
    Movies[specs] = Movies[specs].fillna('')
combined_specs = Movies['title']+' '+Movies['genres']+' '+Movies['keywords']+' '+Movies['tagline']+' '+Movies['cast']+' '+Movies['director']
print(combined_specs)
vectorizer = TfidfVectorizer()
specs_vectors = vectorizer.fit_transform(combined_specs)
print(specs_vectors)
similarity = cosine_similarity(specs_vectors)
print(similarity)
print(similarity.shape)
movie_name = input("Enter your favourite movie name :")
Movie_titles = Movies['title'].tolist()
find_close_match = difflib.get_close_matches(movie_name, Movie_titles)
print(find_close_match)
close_match = find_close_match[0]
print(close_match)
index_value = Movies[Movies.title == close_match]['index'].values[0]
print(index_value)
similarity_score = list(enumerate(similarity[index_value]))
sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True)
print(sorted_similar_movies)
print('Movies we suggest for you: \n')
i = 1
for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = Movies[Movies.index==index]['title'].values[0]
    if (i<10):
                                                   print(i, '.', title_from_index)
                                                   i+=1
