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

# Sentiment Analysis Web App

This project provides a sentiment analysis web application that utilizes Hugging Face's `DistilBERT` transformer model to classify text into positive, negative, or neutral sentiments. The model is fine-tuned for sentiment analysis, and the web interface built with Flask allows users to input text and view sentiment classification results.

## Tech Stack

- **AI Model:** Hugging Face Transformers (DistilBERT)
- **Backend Framework:** Flask
- **Frontend Framework:** HTML, CSS, JavaScript, Bootstrap
- **Environment:** Python 3.x, pip/venv
- **Version Control:** Git and GitHub
- This website was hosted using Ngrok 

# Model Card

**Model Overview**:  
This sentiment analysis model classifies user-inputted text into three categories: positive, negative, or neutral. It uses natural language processing (NLP) techniques to evaluate the emotional tone of the text, providing a quick and automated way to understand sentiment. 

**Intended Use**:  
- **End Users**: For analyzing customer feedback, social media posts, or personal messages.
- **Business Professionals**: To assess product reviews or track brand sentiment.
- **Support Teams**: To prioritize customer interactions based on emotional tone.

**Limitations**:  
- **Bias**: May exhibit biases due to cultural or demographic imbalances in training data.
- **Accuracy**: Struggles with sarcasm, irony, or highly nuanced emotions.
- **Context Sensitivity**: Limited in understanding text context, especially with slang or domain-specific language.

**Ethical Considerations**:  
- Ensure user data privacy and prevent malicious uses such as surveillance or manipulation.
- Maintain transparency about the model's capabilities and limitations.

**Recommendations**:  
- Regular updates and monitoring to mitigate biases and improve accuracy.
- Human oversight in high-stakes applications to verify model outputs.

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


## Model Details

### Model Description

- **Developed by:** [https://huggingface.co/bareeraqrsh]
- **Model type:** [sentiment-analysis]
- **Language(s) (NLP):** [English]
- **License:** [MIT]
- **Finetuned from model [distillbert]:** [sistillbert-base-uncased]

## Uses

The sentiment analysis model is intended for use by individuals, businesses, and organizations to classify text as positive, negative, or neutral.

### Direct Use

When used directly without fine-tuning or integration into a larger ecosystem, the sentiment analysis model can serve a range of standalone functions:

1. **Text Classification**: Users can input any text (such as social media posts, product reviews, or personal messages) and receive an instant classification of the sentiment—positive, negative, or neutral.
   
2. **Content Evaluation**: Individuals or businesses can analyze single or batches of text to assess the general tone and emotional undertone of the content. This could be used for personal reflection or to evaluate customer feedback.

3. **Emotional Tone Detection**: The model can be employed to understand the emotional nuances in writing, which can help in content moderation or detecting sarcasm or anger in text.

4. **Quick Sentiment Assessment**: Ideal for quick assessments where deep context or fine-tuning is not necessary, providing an immediate understanding of sentiment in various documents or messages.

In direct use, the model functions as a standalone tool for basic sentiment analysis, providing insights into the emotional tone of text without any additional customization or system integration.

### Out-of-Scope Use

1. **Sensitive Contexts**: Not for mental health assessments, legal, or medical advice.
2. **Malicious Use**: Should not be used for targeting vulnerable individuals or surveillance.
3. **Inaccurate Outputs**: Struggles with sarcasm, irony, and culturally specific language.
4. **Bias**: May produce biased results if training data is unbalanced.
5. **Over-reliance**: Shouldn’t be the sole basis for critical business or emotional decisions.

## Bias, Risks, and Limitations

1. **Bias**:
   - **Cultural and Linguistic Bias**: The model may misinterpret sentiment in texts with specific cultural references, slang, or dialects outside its training data.
   - **Demographic Bias**: If training data is not diverse, the model may show biased sentiment classifications toward certain groups, leading to unfair outcomes.

2. **Risks**:
   - **Misinterpretation of Complex Emotions**: The model may struggle to understand sarcasm, irony, or nuanced emotions, resulting in incorrect sentiment classifications.
   - **Over-simplification**: The model reduces complex emotions to simple categories (positive, negative, neutral), potentially missing subtleties in the text.
   - **Data Privacy**: If personal data is used, there are risks related to privacy and misuse of sensitive information.

3. **Limitations**:
   - **Context Dependence**: The model may not always capture the full context of a text, especially if it relies on short snippets or lacks background information.
   - **Accuracy Variations**: The model's performance can degrade with poorly structured text, slang, or very domain-specific language.
   - **No Emotional Understanding**: The model classifies sentiment based on patterns, not genuine emotional understanding, limiting its ability to interpret human feelings accurately.

### Recommendations

1. **Bias and Limitations Awareness**: Educate users about potential biases (cultural, demographic) and limitations (sarcasm, nuance).
2. **Transparency**: Disclose model limitations and provide feedback channels for improvement.
3. **Privacy and Ethics**: Ensure strong data privacy and ethical guidelines, avoiding misuse.
4. **Human Oversight**: Use human review for high-stakes applications to ensure accuracy and context.
5. **Regular Evaluation**: Continuously monitor and update the model to improve performance and adapt to new data.


##Result

- Web Interface
  
   ![image](https://github.com/user-attachments/assets/b0bd56b8-b460-4854-9792-1d7eb37ac210)

- Result for sample text file
  
   ![image](https://github.com/user-attachments/assets/7999b8bd-d6f5-47d2-aae1-9d2fc0e9da9e)

- Example text input analysis
  
   ![image](https://github.com/user-attachments/assets/ed72456b-7766-408f-a9a9-de0539ecdebd)


## License

This project is licensed under the MIT License.


## References

https://huggingface.co/blog/sentiment-analysis-python

