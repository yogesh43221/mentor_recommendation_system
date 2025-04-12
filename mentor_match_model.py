
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# Load data
mentors = pd.read_csv('data/mock_mentors.csv')
profiles = pd.read_csv('data/mock_profiles.csv')

# Custom transformer for subject similarity
class SubjectSimilarityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit(self, X, y=None):
        combined = X['preferred_subjects'] + ' ' + X['top_subjects']
        self.vectorizer.fit(combined)
        return self

    def transform(self, X):
        tfidf_pref = self.vectorizer.transform(X['preferred_subjects'])
        tfidf_top = self.vectorizer.transform(X['top_subjects'])
        sim_scores = cosine_similarity(tfidf_pref, tfidf_top).diagonal()
        return sim_scores.reshape(-1, 1)

# Pairwise data creation
def create_pairwise_data(profiles, mentors):
    data = []
    for _, asp in profiles.iterrows():
        for _, men in mentors.iterrows():
            data.append({
                'aspirant_id': asp['aspirant_id'],
                'preferred_subjects': asp['preferred_subjects'],
                'learning_style': asp['learning_style'],
                'target_college': asp['target_college'],
                'preparation_level': asp['preparation_level'],
                'mentor_id': men['mentor_id'],
                'top_subjects': men['top_subjects'],
                'mentor_college': men['mentor_college'],
                'years_of_experience': men['years_of_experience'],
                'mentor_availability': men['mentor_availability'],
                'mentoring_style': men['mentoring_style'],
                'mentee_feedback_rating': men['mentee_feedback_rating'],
                'style_match': int(asp['learning_style'] == men['mentoring_style']),  # compatibility
                'college_match': int(asp['target_college'] == men['mentor_college'])  # compatibility
            })
    return pd.DataFrame(data)

# Create dataset
data = create_pairwise_data(profiles, mentors)

# Add similarity score
sim_transformer = SubjectSimilarityTransformer()
sim_transformer.fit(data)
data['similarity_score'] = sim_transformer.transform(data)

# Get unique similarity scores and calculate bin edges
unique_scores = data['similarity_score'].nunique()

# Adjust bins and labels based on the number of unique scores
if unique_scores >= 3:
    bins = [data['similarity_score'].min(), data['similarity_score'].quantile(0.33), data['similarity_score'].quantile(0.66), data['similarity_score'].max()]
    labels = [0, 1, 2]
elif unique_scores == 2:
    bins = [data['similarity_score'].min(), data['similarity_score'].max()]
    labels = [0]
else:
    bins = [data['similarity_score'].min(), data['similarity_score'].max()]
    labels = [0]

# Ensure bins are unique and that there is one less label than bin edges
bins = sorted(set(bins))

# Adjust labels if the number of labels is not one less than the number of bins
if len(bins) > len(labels) + 1:
    bins = bins[:-1] #remove last bin.
elif len(bins) == len(labels):
    labels = labels[:-1]

# Apply pd.cut to categorize based on similarity score
data['similarity_class'] = pd.cut(data['similarity_score'], bins=bins, labels=labels, include_lowest=True).astype(int)


# Define features and target
X = data.drop(columns=['aspirant_id', 'mentor_id', 'similarity_score'])
y = data['similarity_class']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
categorical_cols = ['learning_style', 'target_college', 'mentor_college', 'mentor_availability', 'mentoring_style', 'preparation_level']
numerical_cols = ['years_of_experience', 'mentee_feedback_rating', 'style_match', 'college_match']

preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
    ('num', 'passthrough', numerical_cols)
])

# Model pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

# Hyperparameter tuning (optional)
param_grid = {
    'classifier__learning_rate': [0.05, 0.1],
    'classifier__max_iter': [100, 200],
    'classifier__max_depth': [3, 6]
}

grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy')
grid.fit(X_train, y_train)

# Evaluation
y_pred = grid.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0)) # add zero_division=0

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Prediction on sample input
input_data = {
    'preferred_subjects': 'English|Logical Reasoning',
    'learning_style': 'Visual',
    'target_college': 'National Law School of India University',
    'preparation_level': 'Beginner',
    'top_subjects': 'English|Logical Reasoning',
    'mentor_college': 'National Law University Delhi',
    'mentor_availability': 'Available',
    'mentoring_style': 'Structured Approach',
    'years_of_experience': 3,
    'mentee_feedback_rating': 4.6,
    'style_match': int('Visual' == 'Structured Approach'),
    'college_match': int('National Law School of India University' == 'National Law University Delhi')
}

input_df = pd.DataFrame([input_data])
probs = grid.predict_proba(input_df)

# Check if class 2 exists in the model's predicted classes
if 2 in grid.best_estimator_.classes_:
    class_2_index = list(grid.best_estimator_.classes_).index(2)
    high_match_probs = probs[:, class_2_index]
else:
    high_match_probs = np.zeros(probs.shape[0])  # If class 2 doesn't exist, return 0s

print(f"Predicted Similarity Class Probabilities (Class 2 if available): {high_match_probs}")
