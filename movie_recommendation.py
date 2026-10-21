import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.DataFrame({
    "title": [
        "Inception", "Interstellar", "The Dark Knight", "Avengers",
        "Iron Man", "Titanic", "The Notebook", "The Matrix",
        "Gravity", "Avatar", "Joker", "Doctor Strange", "Spider-Man"
    ],
    "genre": [
        "Sci-Fi Thriller", "Sci-Fi Adventure", "Action Crime",
        "Action Superhero", "Action Sci-Fi", "Romance Drama",
        "Romance Drama", "Sci-Fi Action", "Sci-Fi Drama",
        "Sci-Fi Adventure", "Crime Drama", "Fantasy Action",
        "Action Superhero"
    ],
    "description": [
        "Dreams and mind bending science fiction",
        "Space travel and exploration",
        "Batman fights crime in Gotham",
        "Superheroes save the world",
        "A genius builds a powerful suit",
        "A love story on a ship",
        "A romantic story about lasting love",
        "A hacker discovers a simulated reality",
        "Astronauts struggle to survive in space",
        "An adventure on an alien planet",
        "A troubled man becomes a criminal",
        "A surgeon learns mystical powers",
        "A young hero fights crime"
    ]
})

movies["content"] = movies["genre"] + " " + movies["description"]

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(movies["content"])
similarity = cosine_similarity(tfidf_matrix)

def recommend_movie(title):
    matches = movies.index[movies["title"].str.lower() == title.lower()]

    if len(matches) == 0:
        print("Movie not found. Please try again.")
        return

    index = matches[0]
    scores = list(enumerate(similarity[index]))
    scores = [item for item in scores if item[0] != index]
    scores.sort(key=lambda x: x[1], reverse=True)

    print("\nRecommended Movies:")
    for movie_index, score in scores[:5]:
        print(
            movies.iloc[movie_index]["title"],
            "- Similarity:",
            round(score, 2)
        )

print("Available Movies:")
for title in movies["title"]:
    print("-", title)

movie_name = input("\nEnter a movie title: ").strip()
recommend_movie(movie_name)
