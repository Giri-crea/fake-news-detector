"""
Fake News Detection - Flask API
REST API endpoint for detecting fake news
"""

from flask import Flask, render_template, request, jsonify
import joblib
import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class NewsDetector:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.load_models()
    
    def load_models(self):
        """Load pre-trained model and vectorizer"""
        try:
            self.classifier = joblib.load('model.pkl')
            self.vectorizer = joblib.load('vectorizer.pkl')
            self.model_loaded = True
            print("Models loaded successfully!")
        except FileNotFoundError:
            self.model_loaded = False
            print("Warning: Model files not found. Train the model first using train_model.py")
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'<.*?>', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        text = ' '.join(text.split())
        
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        
        return ' '.join(words)
    
    def predict(self, text):
        """Predict if news is fake or real"""
        if not self.model_loaded:
            return {
                'error': 'Model not loaded. Please train the model first.',
                'prediction': None,
                'confidence': None
            }
        
        processed = self.preprocess_text(text)
        vectorized = self.vectorizer.transform([processed])
        prediction = self.classifier.predict(vectorized)[0]
        confidence = self.classifier.decision_function(vectorized)[0]
        
        return {
            'text': text[:100],
            'prediction': 'FAKE NEWS' if prediction == 1 else 'REAL NEWS',
            'confidence': float(abs(confidence)),
            'label': int(prediction),
            'error': None
        }


detector = NewsDetector()


@app.route('/')
def index():
    """Render the home page"""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for fake news detection"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text or len(text.strip()) == 0:
            return jsonify({'error': 'Text field is required'}), 400
        
        if len(text) > 10000:
            return jsonify({'error': 'Text too long. Maximum 10000 characters'}), 400
        
        result = detector.predict(text)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/info', methods=['GET'])
def info():
    """Get API information"""
    return jsonify({
        'name': 'Fake News Detection API',
        'version': '1.0.0',
        'model_loaded': detector.model_loaded,
        'endpoints': {
            'POST /api/predict': 'Detect if text is fake news',
            'GET /api/info': 'Get API information'
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': detector.model_loaded})


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
