"""
Fake News Detection - Model Training Script
Uses TF-IDF vectorization with Passive Aggressive Classifier
"""

import pandas as pd
import numpy as np
import re
import string
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

class FakeNewsModelTrainer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_df=0.7, max_features=10000)
        self.classifier = SGDClassifier(loss='hinge', penalty=None, max_iter=1000, random_state=42)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove stopwords and lemmatize
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        
        return ' '.join(words)
    
    def create_sample_dataset(self):
        """Create a sample dataset for demonstration"""
        fake_news = [
            "BREAKING: Scientists discover that the moon is made of cheese!",
            "Government secretly controls weather using chemtrails",
            "Aliens landed in New York and met with world leaders",
            "Drinking bleach cures all diseases according to doctors",
            "Facebook will start charging users $10 per month",
            "Celebrity death hoax spreads across social media",
            "Miracle pill helps you lose 50 pounds in one week",
            "Secret society controls all world governments",
            "Vaccines contain microchips for mind control",
            "Earth is actually flat according to leaked NASA documents",
        ]
        
        real_news = [
            "Stock market closes higher amid positive economic data",
            "New climate report shows rising global temperatures",
            "Scientists develop new treatment for cancer patients",
            "Government announces infrastructure spending plan",
            "Tech companies report quarterly earnings growth",
            "Research shows benefits of regular exercise",
            "Central bank maintains interest rates unchanged",
            "New study examines effects of social media on teens",
            "Electric vehicle sales continue to grow worldwide",
            "International summit addresses trade agreements",
        ]
        
        data = {
            'text': fake_news + real_news,
            'label': [1] * len(fake_news) + [0] * len(real_news)
        }
        
        return pd.DataFrame(data)
    
    def train(self, df=None, text_column='text', label_column='label'):
        """Train the fake news detection model"""
        if df is None:
            print("Using sample dataset for demonstration.")
            df = self.create_sample_dataset()
        
        print(f"Dataset size: {len(df)} samples")
        print(f"Fake news: {sum(df[label_column] == 1)}")
        print(f"Real news: {sum(df[label_column] == 0)}")
        
        # Preprocess text
        print("\nPreprocessing text...")
        df['processed_text'] = df[text_column].apply(self.preprocess_text)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], 
            df[label_column], 
            test_size=0.2, 
            random_state=42
        )
        
        # Vectorize text
        print("Vectorizing text...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Train classifier
        print("Training classifier...")
        self.classifier.fit(X_train_tfidf, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n{'='*50}")
        print(f"Model Accuracy: {accuracy:.2%}")
        print(f"{'='*50}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
        
        return accuracy
    
    def save_model(self, model_path='model', vectorizer_path='vectorizer'):
        """Save trained model and vectorizer"""
        joblib.dump(self.classifier, f'{model_path}.pkl')
        joblib.dump(self.vectorizer, f'{vectorizer_path}.pkl')
        print(f"\nModel saved to {model_path}.pkl")
        print(f"Vectorizer saved to {vectorizer_path}.pkl")
    
    def load_model(self, model_path='model', vectorizer_path='vectorizer'):
        """Load trained model and vectorizer"""
        self.classifier = joblib.load(f'{model_path}.pkl')
        self.vectorizer = joblib.load(f'{vectorizer_path}.pkl')
        print("Model loaded successfully!")
    
    def predict(self, text):
        """Predict if news is fake or real"""
        processed = self.preprocess_text(text)
        vectorized = self.vectorizer.transform([processed])
        prediction = self.classifier.predict(vectorized)[0]
        confidence = self.classifier.decision_function(vectorized)[0]
        
        return {
            'prediction': 'FAKE' if prediction == 1 else 'REAL',
            'confidence': abs(confidence),
            'label': int(prediction)
        }


def main():
    """Main function to train and save the model"""
    trainer = FakeNewsModelTrainer()
    trainer.train()
    trainer.save_model()
    
    print("\n" + "="*50)
    print("Testing Predictions:")
    print("="*50)
    
    test_texts = [
        "Scientists announce breakthrough in renewable energy research",
        "SHOCKING: Politicians are secretly lizard people from outer space!"
    ]
    
    for text in test_texts:
        result = trainer.predict(text)
        print(f"\nText: {text[:60]}...")
        print(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.2f})")


if __name__ == "__main__":
    main()
