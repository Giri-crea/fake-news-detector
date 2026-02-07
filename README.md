# Fake News Detection System

A complete AI-powered system for detecting fake news using machine learning. Includes model training, REST API, and web interface.

## Features

- **Machine Learning Model**: Uses TF-IDF vectorization with Passive Aggressive Classifier
- **REST API**: Flask-based API for integration with other applications
- **Web Interface**: Beautiful, responsive web UI for interactive news analysis
- **Text Preprocessing**: Comprehensive text cleaning and normalization
- **Confidence Scoring**: Provides confidence levels for predictions

## Project Structure

```
fake_news_detector/
├── requirements.txt          # Python dependencies
├── train_model.py           # Model training script
├── app.py                   # Flask API application
├── templates/
│   └── index.html          # Web interface
├── README.md               # This file
├── model.pkl               # Trained model (generated after training)
└── vectorizer.pkl          # TF-IDF vectorizer (generated after training)
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup Steps

1. Create the project directory:
```bash
mkdir fake_news_detector
cd fake_news_detector
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model:
```bash
python train_model.py
```

This will:
- Create a sample dataset with fake and real news
- Train the classifier
- Save `model.pkl` and `vectorizer.pkl`
- Display accuracy metrics

## Usage

### Option 1: Web Interface (Recommended for Users)

Start the Flask application:
```bash
python app.py
```

Then open your browser and navigate to:
```
http://localhost:5000
```

Features:
- Paste news text into the input area
- Click "Analyze News" to get predictions
- View confidence scores and prediction results
- Responsive design works on desktop and mobile

### Option 2: REST API (For Developers)

Start the Flask application:
```bash
python app.py
```

#### API Endpoints

**POST /api/predict**
- Detect if text is fake news
- Request body:
```json
{
    "text": "Your news article here..."
}
```
- Response:
```json
{
    "text": "Your news article here...",
    "prediction": "FAKE NEWS",
    "confidence": 0.85,
    "label": 1,
    "error": null
}
```

**GET /api/info**
- Get API information
- Response includes version, endpoints, and model status

**GET /health**
- Health check endpoint
- Returns API health status

#### Example API Calls

Using curl:
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Scientists discover breakthrough in renewable energy"}'
```

Using Python:
```python
import requests

url = "http://localhost:5000/api/predict"
data = {"text": "Your news text here"}
response = requests.post(url, json=data)
result = response.json()
print(result)
```

Using JavaScript:
```javascript
fetch('/api/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text: 'Your news text here'})
})
.then(r => r.json())
.then(data => console.log(data));
```

### Option 3: Python Script

Use the trained model in your Python code:
```python
from train_model import FakeNewsModelTrainer

trainer = FakeNewsModelTrainer()
trainer.load_model()

result = trainer.predict("Your news text here")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2f}")
```

## Model Details

### Algorithm
- **Vectorizer**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Classifier**: Passive Aggressive Classifier
- **Max Features**: 10,000 most important words
- **Test Split**: 80% training, 20% testing

### Text Preprocessing
1. Lowercase conversion
2. URL removal
3. HTML tag removal
4. Punctuation removal
5. Number removal
6. Stopword removal
7. Lemmatization

### Performance
The model achieves high accuracy on the sample dataset. To improve accuracy:
- Use a larger, real-world dataset
- Fine-tune hyperparameters
- Experiment with different algorithms
- Use ensemble methods

## Training with Custom Data

To train with your own dataset:

```python
import pandas as pd
from train_model import FakeNewsModelTrainer

# Load your data
df = pd.read_csv('your_dataset.csv')

# Expected format:
# - 'text' column: news article text
# - 'label' column: 1 for fake, 0 for real

trainer = FakeNewsModelTrainer()
trainer.train(df, text_column='text', label_column='label')
trainer.save_model()
```

## Dataset Format

Your CSV should have at least these columns:
- `text`: The news article content
- `label`: 1 for fake news, 0 for real news

Example:
```csv
text,label
"Breaking: Scientists discover moon is made of cheese",1
"Stock market rises amid positive economic data",0
```

## Troubleshooting

### Model files not found
- Make sure you've run `train_model.py` first
- Check that `model.pkl` and `vectorizer.pkl` exist in the same directory

### NLTK data not found
- The script downloads required data automatically
- If issues persist, manually download:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

### Port 5000 already in use
- Change the port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Low accuracy
- The sample dataset is small for demonstration
- Use a real-world dataset for production
- Consider using pre-trained models like BERT for better results

## Deployment

### Using Gunicorn (Production)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

### Using Docker
Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "app:app"]
```

Build and run:
```bash
docker build -t fake-news-detector .
docker run -p 8000:8000 fake-news-detector
```

## Limitations

1. **Sample Data**: The included dataset is small and synthetic
2. **Language**: Currently optimized for English
3. **Context**: Relies on text content, not external fact-checking
4. **Dynamic News**: May not detect newly emerging fake news patterns
5. **Bias**: Model performance depends on training data quality

## Future Improvements

- Integration with fact-checking APIs
- Multi-language support
- Deep learning models (LSTM, BERT)
- Real-time news feed monitoring
- User feedback system for model improvement
- Visualization of important features

## License

This project is open source and available for educational and research purposes.

## Support

For issues, questions, or contributions, please refer to the project documentation or contact the development team.

## Disclaimer

This tool is designed for educational and research purposes. It should not be the sole basis for determining news authenticity. Always verify important information from multiple reliable sources.
