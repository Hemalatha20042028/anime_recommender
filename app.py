# app.py
from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

app = Flask(__name__)

# Check if precomputed files exist
if not os.path.exists('animes.pkl') or not os.path.exists('similarity.pkl'):
    # Load only first 2000 rows
    anime = pd.read_csv("anime.csv", nrows=2000)
    
    # Handle missing values
    anime['genre'].fillna('Unknown', inplace=True)
    anime['type'].fillna('Unknown', inplace=True)
    anime['rating'].fillna(anime['rating'].median(), inplace=True)
    
    # Create tags column
    anime['tags'] = anime['genre'] + " ," + anime['type'] + "," + anime['episodes'].astype(str)
    
    # Create new dataframe
    new_df = anime[['anime_id', 'name', 'tags']]
    
    # Vectorize tags
    cv = CountVectorizer(max_features=2000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    
    # Compute similarity matrix
    similarity = cosine_similarity(vectors)
    
    # Save to files
    pickle.dump(new_df, open('animes.pkl', 'wb'))
    pickle.dump(similarity, open('similarity.pkl', 'wb'))
    print("Created and saved data files")
else:
    # Load precomputed data
    new_df = pickle.load(open('animes.pkl', 'rb'))
    similarity = pickle.load(open('similarity.pkl', 'rb'))
    print("Loaded precomputed data files")

# Get list of anime names for search suggestions
anime_names = sorted(new_df['name'].tolist())

@app.route('/')
def home():
    return render_template('index.html', anime_names=anime_names)

@app.route('/recommend', methods=['POST'])
def recommend():
    anime_name = request.form['anime_name']
    
    if anime_name not in new_df['name'].values:
        return render_template('error.html',
                               anime_name=anime_name,
                               message="❌ Anime not found in database")
    
    # Find the anime index
    anime_index = new_df[new_df['name'] == anime_name].index[0]
    
    # Get similarity scores
    distances = similarity[anime_index]
    
    # Get top 5 recommendations (excluding the anime itself)
    animes_list = sorted(
        list(enumerate(distances)),
        key=lambda x: x[1],
        reverse=True
    )[1:6]
    
    recommendations = [new_df.iloc[idx]['name'] for idx, _ in animes_list]
    
    return render_template('recommendations.html',
                           anime_name=anime_name,
                           recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
