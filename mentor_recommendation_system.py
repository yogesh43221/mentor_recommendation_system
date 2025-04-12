import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Load mock data (from CSVs)
def load_data():
    aspirants = pd.read_csv('data/mock_profiles.csv')
    mentors = pd.read_csv('data/mock_mentors.csv')

    # Ensure numeric conversion
    mentors['years_of_experience'] = pd.to_numeric(mentors['years_of_experience'], errors='coerce')
    mentors['mentee_feedback_rating'] = pd.to_numeric(mentors['mentee_feedback_rating'], errors='coerce')

    return aspirants, mentors

# 2. Preprocess and prepare data for similarity calculation
def preprocess_data(aspirants, mentors):
    # Define feature groups
    categorical_features = ['learning_style']
    numerical_features = []  # Leave it empty since 'preparation_level' isn't in the mentors data

    # Vectorize text fields
    tfidf = TfidfVectorizer()
    aspirants_tfidf = tfidf.fit_transform(aspirants['preferred_subjects'])
    mentors_tfidf = tfidf.transform(mentors['top_subjects'])

    # Only select columns required for transformer
    aspirants_trans_input = aspirants[categorical_features]
    mentors_trans_input = mentors[categorical_features]

    # ColumnTransformer for categorical
    transformer = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(), categorical_features)]
    )

    # Fit-transform on aspirants, transform on mentors
    aspirants_features = transformer.fit_transform(aspirants_trans_input)
    mentors_features = transformer.transform(mentors_trans_input)

    # Combine TF-IDF with transformed features
    aspirants_final = np.hstack([aspirants_tfidf.toarray(), aspirants_features])  # No need to call .toarray() on aspirants_features
    mentors_final = np.hstack([mentors_tfidf.toarray(), mentors_features])  # No need to call .toarray() on mentors_features

    return aspirants_final, mentors_final



# 3. Calculate cosine similarity between aspirant and mentor features
def calculate_similarity(aspirant_vector, mentor_vector):
    return cosine_similarity([aspirant_vector], [mentor_vector])[0][0]

# 4. Recommend mentors for a given aspirant based on similarity
def recommend_mentors(aspirant_id, aspirants, mentors, aspirants_final, mentors_final, top_n=3):
    # Retrieve the specific aspirant by ID
    aspirant_index = aspirants[aspirants['aspirant_id'] == aspirant_id].index[0]
    aspirant_vector = aspirants_final[aspirant_index]

    mentor_scores = []
    for i, mentor in mentors.iterrows():
        mentor_vector = mentors_final[i]
        score = calculate_similarity(aspirant_vector, mentor_vector)
        mentor_scores.append((mentor['mentor_id'], score))

    sorted_mentors = sorted(mentor_scores, key=lambda x: x[1], reverse=True)[:top_n]
    return sorted_mentors

# 5. Main interface
def main():
    aspirants, mentors = load_data()
    aspirants_final, mentors_final = preprocess_data(aspirants, mentors)

    # Input: Aspirant ID
    aspirant_id = int(input("Enter Aspirant ID: ").strip())

    # Recommendations
    recommendations = recommend_mentors(aspirant_id, aspirants, mentors, aspirants_final, mentors_final)

    print(f"\nTop {len(recommendations)} Recommended Mentors for Aspirant {aspirant_id}:")
    for i, (mentor_id, score) in enumerate(recommendations, 1):
        print(f"{i}. Mentor {mentor_id} - Similarity Score: {score:.2f}")

if __name__ == "__main__":
    main()
