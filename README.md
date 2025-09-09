---
license: mit
datasets:
- Sp1786/multiclass-sentiment-analysis-dataset
language:
- en
metrics:
- Accuracy
- f1-score
base_model:
- distilbert/distilbert-base-uncased
library_name: transformers
---

# Sentiment Analysis Web Application

A comprehensive sentiment analysis tool that classifies text into positive, negative, or neutral sentiments using a fine-tuned DistilBERT transformer model. [1](#2-0)  The application features both single text analysis and batch file processing through an intuitive web interface.

## 🚀 Features

- **Real-time Sentiment Analysis**: Classify individual text inputs instantly
- **Batch Processing**: Upload `.txt` or `.csv` files for bulk sentiment analysis
- **Three-Class Classification**: Negative, Neutral, and Positive sentiment detection
- **Confidence Scores**: Get probability scores for each prediction
- **Responsive Web Interface**: Bootstrap-powered UI that works on all devices
- **Multiple Deployment Options**: Development (Jupyter + ngrok) and production (Render.com) ready

## 🛠️ Tech Stack

- **AI Model**: Hugging Face Transformers (DistilBERT) [2](#2-1) 
- **Backend**: Flask web framework [3](#2-2) 
- **Frontend**: HTML, CSS, JavaScript, Bootstrap [4](#2-3) 
- **Environment**: Python 3.x [5](#2-4) 
- **Deployment**: Render.com, ngrok tunneling [6](#2-5) 

## 📊 Model Information

### Model Details
- **Base Model**: `distilbert-base-uncased` [7](#2-6) 
- **Model Repository**: [bareeraqrsh/Sentiment-analysis-tool](https://huggingface.co/bareeraqrsh/Sentiment-analysis-tool) [8](#2-7) 
- **Dataset**: [Sp1786/multiclass-sentiment-analysis-dataset](https://huggingface.co/datasets/Sp1786/multiclass-sentiment-analysis-dataset) [9](#2-8) 
- **License**: MIT [10](#2-9) 
- **Language**: English [11](#2-10) 

### Performance Metrics
- **Accuracy**: Tracked via wandb.ai [12](#2-11) 
- **F1-Score**: Multi-class evaluation [13](#2-12) 

## 🚀 Quick Start

### Prerequisites
```bash
pip install flask transformers torch numpy
```

### Running the Application

#### Option 1: Development (Jupyter Notebook)
1. Open `flask_interface.ipynb` in Google Colab or Jupyter
2. Install dependencies and set up ngrok authentication
3. Run all cells to start the Flask server with ngrok tunneling

#### Option 2: Production Deployment
Deploy directly to Render.com using the included configuration files.

### Model Loading
The application automatically loads the pre-trained model and tokenizer:

```python
MODEL_NAME = "bareeraqrsh/Sentiment-analysis-tool"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
LABELS = ["Negative", "Neutral", "Positive"]
```

## 💻 Usage

### Web Interface
1. **Single Text Analysis**: Enter text in the textarea and click "Analyze"
2. **Batch Processing**: Upload a `.txt` or `.csv` file with one text per line
3. **Results**: View predictions with confidence scores in an organized table

### API Endpoint
- **Route**: `/` (GET, POST)
- **Methods**: Form submission with text input or file upload
- **Response**: HTML page with sentiment predictions and confidence scores

## 📁 Project Structure

```
sentiment-analysis-tool/
├── README.md                          # Project documentation
├── Sentiment_analysis_model.ipynb     # Model training notebook
├── flask_interface.ipynb              # Web application (development)
├── requirements.txt                   # Python dependencies
├── .render.yaml                       # Production deployment config
└── model_artifacts/                   # Trained model files (Git LFS)
    ├── model.safetensors              # Model weights (~267MB)
    ├── config.json                    # Model configuration
    ├── tokenizer_config.json          # Tokenizer settings
    └── vocab.txt                      # Vocabulary file
```

## 🎯 Use Cases

### Direct Applications
- **Customer Feedback Analysis**: Analyze product reviews and customer comments [14](#2-13) 
- **Social Media Monitoring**: Track brand sentiment across platforms
- **Content Moderation**: Detect negative sentiment in user-generated content
- **Support Ticket Prioritization**: Identify urgent issues based on emotional tone

### Business Applications
- Market research and competitor analysis
- Employee feedback evaluation
- Campaign effectiveness measurement
- Risk assessment in communications

## ⚠️ Limitations & Considerations

### Model Limitations
- **Sarcasm & Irony**: Limited ability to detect nuanced expressions [15](#2-14) 
- **Context Sensitivity**: May struggle with domain-specific language or slang [16](#2-15) 
- **Cultural Bias**: Potential biases from training data demographics [17](#2-16) 

### Ethical Guidelines
- Ensure user data privacy and consent [18](#2-17) 
- Avoid surveillance or manipulation applications [19](#2-18) 
- Implement human oversight for critical decisions [20](#2-19) 

### Recommendations
- Regular model evaluation and bias monitoring [21](#2-20) 
- Human review for high-stakes applications
- Transparent communication of model capabilities and limitations

## 🔧 Development

### Training Pipeline
The model training process is documented in `Sentiment_analysis_model.ipynb`, which includes:
- Data loading and preprocessing from Hugging Face datasets [22](#2-21) 
- DistilBERT fine-tuning with custom classification head
- Model evaluation and performance tracking via wandb.ai [23](#2-22) 
- Model artifact generation and storage

### Deployment Options
1. **Development**: Jupyter notebook with ngrok tunneling for testing
2. **Production**: Render.com deployment with Gunicorn WSGI server
3. **Local**: Standard Flask development server

## 📈 Performance Monitoring

The training process includes comprehensive evaluation metrics tracked through wandb.ai, providing insights into:
- Training and validation accuracy
- Loss curves and convergence patterns
- F1-scores for multi-class classification
- Model performance across different sentiment categories

## Dataset and training Details

Dataset loaded from dataset library in huggingface hub contains splits for dataset into training, validation and testing datasets.

- Link to the dataset used:

   https://huggingface.co/datasets/Sp1786/multiclass-sentiment-analysis-dataset

- HuggingFace repo for this model:

   https://huggingface.co/bareeraqrsh/Sentiment-analysis-tool

- Base model used:

   https://huggingface.co/distilbert/distilbert-base-uncased

   This model is a distilled version of the BERT base model.   

- The run and evaluation was visualized using wandb.ai

   ![image](https://github.com/user-attachments/assets/d7950fa2-bc79-41dc-86f4-bd12f7248445)


   ![image](https://github.com/user-attachments/assets/48f49c43-8684-4952-8743-23007b305cdb)


##Result

- Web Interface
  
   ![image](https://github.com/user-attachments/assets/b0bd56b8-b460-4854-9792-1d7eb37ac210)

- Result for sample text file
  
   ![image](https://github.com/user-attachments/assets/7999b8bd-d6f5-47d2-aae1-9d2fc0e9da9e)

- Example text input analysis
  
   ![image](https://github.com/user-attachments/assets/ed72456b-7766-408f-a9a9-de0539ecdebd)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License. [24](#2-23)  See the LICENSE file for details.

## 🔗 References

- [Hugging Face Sentiment Analysis Guide](https://huggingface.co/blog/sentiment-analysis-python) [25](#2-24) 
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
- [Transformers Library Documentation](https://huggingface.co/docs/transformers/)

## 📞 Support

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Check the existing documentation and wiki pages
- Review the model card for detailed usage guidelines

**Note**: This tool is designed for general sentiment analysis tasks. For specialized domains or critical applications, consider additional fine-tuning or human validation of results.
