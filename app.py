from flask import Flask, request, jsonify
from flask_cors import CORS
from model import AssessmentRecommender
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
MODEL_PATH = 'assessment_recommender.pkl'
recommender = None

def load_model():
    global recommender
    if os.path.exists(MODEL_PATH):
        recommender = AssessmentRecommender.load_model(MODEL_PATH)
    else:
        recommender = AssessmentRecommender()
        recommender.fit('shl_assessments_rag.csv')
        recommender.save_model(MODEL_PATH)

@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    try:
        # Get job description from request
        data = request.get_json()
        job_description = data.get('job_description', '')
        
        if not job_description:
            return jsonify({
                'error': 'No job description provided'
            }), 400
        
        # Get recommendations
        recommendations = recommender.get_recommendations(job_description)
        
        return jsonify({
            'success': True,
            'recommendations': recommendations
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': recommender is not None
    })
    
load_model()

if __name__ == '__main__':
    # Load the model before starting the server
    #load_model()
    app.run(debug=True, port=5000) 