# Sentiment Analysis Web App

This project provides a sentiment analysis web application that utilizes Hugging Face's `DistilBERT` transformer model to classify text into positive, negative, or neutral sentiments. The model is fine-tuned for sentiment analysis, and the web interface built with Flask allows users to input text and view sentiment classification results.

## Tech Stack

- **AI Model:** Hugging Face Transformers (DistilBERT)
- **Backend Framework:** Flask
- **Frontend Framework:** HTML, CSS, JavaScript
- **Environment:** Python 3.x, pip/venv
- **Version Control:** Git and GitHub

## Setup Instructions

Follow these steps to set up the project on your local machine:

### 1. Clone the repository
First, clone the repository to your local machine using Git:

```bash
git clone https://github.com/yourusername/sentiment-analysis-app.git
cd sentiment-analysis-app
```

### 2. Set up a virtual environment
Create and activate a Python virtual environment to manage dependencies:

#### On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should include the following dependencies:

```txt
Flask==2.2.2
transformers==4.30.0
torch==2.0.0
```

### 4. Fine-tune DistilBERT (optional)
If you want to fine-tune the model yourself, you can follow the Hugging Face tutorial to fine-tune DistilBERT on a sentiment analysis dataset. Otherwise, you can use the pre-trained model directly.

### 5. Run the Flask app
To start the Flask server, run the following command:

```bash
python app.py
```

The application will be accessible at `http://127.0.0.1:5000/` on your local machine.

### 6. Access the app
Open your browser and navigate to `http://127.0.0.1:5000/`. You should see the sentiment analysis web app interface.

### 7. Testing the app
Enter a sentence or a paragraph into the input field and click the "Analyze Sentiment" button. The app will classify the sentiment as either "Positive", "Negative", or "Neutral" and display the result on the webpage.

## Features

- **Text Input:** Users can input any text to analyze its sentiment.
- **Sentiment Classification:** The app classifies the sentiment of the text as Positive, Negative, or Neutral using the fine-tuned DistilBERT model.
- **Web Interface:** A user-friendly HTML form to input text and display results.

## Sample Usage

1. Start the Flask application:
   ```bash
   python app.py
   ```

2. Open your browser and go to `http://127.0.0.1:5000/`.

3. Enter a sentence (e.g., "I love this product!") in the input field.

4. Click the "Analyze Sentiment" button.

5. The sentiment classification (e.g., Positive, Negative, or Neutral) will be displayed below the input.

### Example:

#### Input:
```text
I absolutely hate this service.
```

#### Output:
```text
Sentiment: Negative
```

## License

This project is licensed under the MIT License.
