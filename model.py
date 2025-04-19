import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

class AssessmentRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)
        )
        self.assessments_df = None
        self.tfidf_matrix = None
        
    def preprocess_text(self, text):
        """Clean and preprocess text data."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        # Join tokens back into text
        return ' '.join(tokens)
    
    def fit(self, csv_path):
        """Train the model using the assessment data."""
        # Load and preprocess the data
        self.assessments_df = pd.read_csv(csv_path)
        
        # Create a combined text field for vectorization
        self.assessments_df['combined_features'] = (
            self.assessments_df['Assessment Name'].fillna('') + ' ' +
            self.assessments_df['Job Level'].fillna('')
        )
        
        # Preprocess the combined features
        self.assessments_df['combined_features'] = self.assessments_df['combined_features'].apply(self.preprocess_text)
        
        # Create TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(self.assessments_df['combined_features'])
        
        return self
    
    def get_recommendations(self, job_description, top_n=10):
        """Get top N assessment recommendations for a job description."""
        # Preprocess the input job description
        processed_description = self.preprocess_text(job_description)
        
        # Transform the job description using the fitted vectorizer
        description_vector = self.vectorizer.transform([processed_description])
        
        # Calculate similarity scores
        similarity_scores = cosine_similarity(description_vector, self.tfidf_matrix)
        
        # Get indices of top N similar assessments
        top_indices = similarity_scores[0].argsort()[::-1][:top_n]
        
        # Create recommendations list with similarity scores
        recommendations = []
        for idx in top_indices:
            assessment = self.assessments_df.iloc[idx]
            similarity_score = similarity_scores[0][idx]
            recommendations.append({
                'id': str(idx),
                'title': assessment['Assessment Name'],
                'description': '',  # Add description if available in your dataset
                'category': 'Assessment',  # Add category if available in your dataset
                'duration': assessment['Duration'],
                'skills': [],  # Add skills if available in your dataset
                'benefits': [],  # Add benefits if available in your dataset
                'suitableFor': [],  # Add suitable roles if available in your dataset
                'imageUrl': '',  # Add image URL if available
                'jobLevel': assessment['Job Level'],
                'remoteTestingAvailable': assessment['Remote Testing'] == 'Yes',
                'similarity_score': similarity_score
            })
        
        return recommendations
    
    def save_model(self, filepath):
        """Save the trained model to a file."""
        model_data = {
            'vectorizer': self.vectorizer,
            'assessments_df': self.assessments_df,
            'tfidf_matrix': self.tfidf_matrix
        }
        joblib.dump(model_data, filepath)
    
    @classmethod
    def load_model(cls, filepath):
        """Load a trained model from a file."""
        model = cls()
        model_data = joblib.load(filepath)
        model.vectorizer = model_data['vectorizer']
        model.assessments_df = model_data['assessments_df']
        model.tfidf_matrix = model_data['tfidf_matrix']
        return model

# Train and save the model
if __name__ == "__main__":
    # Initialize and train the model
    recommender = AssessmentRecommender()
    recommender.fit('shl_assessments_rag.csv')
    
    # Save the trained model
    recommender.save_model('assessment_recommender.pkl')
    
    # Test the model
    test_description = """
    Looking for a software engineer with strong programming skills in Python and JavaScript.
    Should have experience in web development and be able to work in a team environment.
    """
    
    recommendations = recommender.get_recommendations(test_description)
    print("\nTest Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['title']}")
        print(f"   Similarity Score: {rec['similarity_score']}")
        print(f"   Job Level: {rec['jobLevel']}")
        print(f"   Duration: {rec['duration']}") 