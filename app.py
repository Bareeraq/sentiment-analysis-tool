# -*- coding: utf-8 -*-
import os
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

app = Flask(__name__)

HF_TOKEN = os.getenv('HF_TOKEN')
if not HF_TOKEN:
    # For Render deployment, this should be set in environment variables
    print("Warning: HF_TOKEN environment variable not set")

# Load model
MODEL_NAME = "bareeraqrsh/Sentiment-analysis-tool"
try:
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=3, 
        token=HF_TOKEN
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, 
        token=HF_TOKEN
    )
    LABELS = ["Negative", "Neutral", "Positive"]
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    tokenizer = None
    LABELS = []

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    text = None
    batch_results = []

    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' in request.files:
            uploaded_file = request.files['file']
            if uploaded_file.filename != '':
                if uploaded_file.filename.endswith(('.txt', '.csv')):
                    try:
                        # Read the uploaded file
                        content = uploaded_file.read().decode('utf-8')
                        texts = content.splitlines()

                        # Analyze each line of text
                        for line in texts:
                            if line.strip():  # Skip empty lines
                                if model and tokenizer:
                                    inputs = tokenizer(line, return_tensors="pt", padding=True, truncation=True, max_length=512)
                                    with torch.no_grad():
                                        outputs = model(**inputs)
                                        logits = outputs.logits
                                        probabilities = torch.softmax(logits, dim=-1).numpy().flatten()
                                        predicted_label = LABELS[np.argmax(probabilities)]
                                        confidence = probabilities[np.argmax(probabilities)]
                                    batch_results.append((line, predicted_label, round(confidence, 2)))
                                else:
                                    batch_results.append((line, "Model not loaded", 0.0))
                    except Exception as e:
                        prediction = f"Error processing file: {str(e)}"
                else:
                    prediction = "Please upload a valid .txt or .csv file."
        
        # Handle text input from the textarea
        text = request.form.get('text', '')
        if text and model and tokenizer:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1).numpy().flatten()
            predicted_label = LABELS[np.argmax(probabilities)]
            confidence = probabilities[np.argmax(probabilities)]
            prediction = f"Prediction: {predicted_label}, Confidence: {confidence:.2f}"

    return render_template('index.html', text=text, prediction=prediction, batch_results=batch_results)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    if not model or not tokenizer:
        return jsonify({'error': 'Model not loaded'}), 500
        
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
        
    text = data.get('text', '')

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1).numpy().flatten()

    predicted_label = LABELS[np.argmax(probabilities)]
    confidence = float(probabilities[np.argmax(probabilities)])

    return jsonify({
        'text': text,
        'sentiment': predicted_label,
        'confidence': confidence,
        'probabilities': {label: float(prob) for label, prob in zip(LABELS, probabilities)}
    })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.getenv('FLASK_DEBUG', 'False').lower() == 'true')