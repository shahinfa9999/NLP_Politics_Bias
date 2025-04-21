import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from transformers import AutoTokenizer, AutoModel
import torch
import spacy

# Load the saved models and resources
print("Loading models and resources...")
vectorizer = joblib.load('vectorizer.pkl')
lda_model = joblib.load('lda_model.pkl')
rf_model = joblib.load('random_forest_model.pkl')

# Load XGBoost model
import xgboost as xgb
xgb_model = xgb.XGBClassifier()
xgb_model.load_model('xgboost_model.json')

# Load Neural Network model
nn_model = tf.keras.models.load_model('neural_network_model.h5')

# Load BERT model for embeddings
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModel.from_pretrained("roberta-base")

# Load spaCy for preprocessing
nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
    """Preprocess text by removing source markers and lemmatizing"""
    # Replace source markers
    text = (text
        .replace("(CNN)", "SOURCE")
        .replace("CNN.com", "SOURCE")
        .replace("CNN", "SOURCE")
        .replace("Daily Mail", "SOURCE")
        .replace("Rueters", "SOURCE")
    )
    
    # Lemmatize using spaCy
    doc = nlp(text)
    lemmatized = [token.lemma_ for token in doc if not token.is_punct and not token.is_stop]
    return ' '.join(lemmatized)

def get_bert_embeddings(text):
    """Get BERT embeddings for a single text"""
    inputs = tokenizer([text], padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # Use CLS token embedding
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

def classify_article(text, model_type='ensemble'):
    """
    Classify a news article as politically 'left' or 'right'
    
    Parameters:
    text (str): The news article text
    model_type (str): The model to use ('rf', 'xgb', 'nn', or 'ensemble')
    
    Returns:
    str: The political bias classification ('left' or 'right')
    float: Confidence score (probability for the predicted class)
    """
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Create document-term matrix
    X_counts = vectorizer.transform([processed_text])
    
    # Transform to topic distribution
    X_topics = lda_model.transform(X_counts)
    
    # Get BERT embeddings
    X_bert = get_bert_embeddings(processed_text)
    
    # Combine features
    X_combined = np.hstack([X_topics, X_bert])
    
    # Make prediction based on specified model
    if model_type == 'rf':
        # Random Forest
        prediction = rf_model.predict(X_combined)[0]
        confidence = np.max(rf_model.predict_proba(X_combined)[0])
    elif model_type == 'xgb':
        # XGBoost
        prediction = xgb_model.predict(X_combined)[0]
        proba = xgb_model.predict_proba(X_combined)[0]
        confidence = proba[prediction]
    elif model_type == 'nn':
        # Neural Network
        pred_prob = nn_model.predict(X_combined)[0][0]
        prediction = 1 if pred_prob > 0.5 else 0
        confidence = pred_prob if prediction == 1 else 1 - pred_prob
    else:
        # Ensemble: combine all models (majority vote)
        rf_pred = rf_model.predict(X_combined)[0]
        xgb_pred = xgb_model.predict(X_combined)[0]
        nn_pred = 1 if nn_model.predict(X_combined)[0][0] > 0.5 else 0
        
        # Count votes
        votes = [rf_pred, xgb_pred, nn_pred]
        prediction = max(set(votes), key=votes.count)
        
        # Average confidence
        rf_conf = rf_model.predict_proba(X_combined)[0][prediction]
        xgb_conf = xgb_model.predict_proba(X_combined)[0][prediction] 
        nn_conf = nn_model.predict(X_combined)[0][0]
        nn_conf = nn_conf if prediction == 1 else 1 - nn_conf
        confidence = (rf_conf + xgb_conf + nn_conf) / 3
    
    # Map numeric prediction to label
    label = 'right' if prediction == 1 else 'left'
    
    return label, confidence

### AI anf history summary functions 
import os
from openai import OpenAI
from dotenv import load_dotenv
from prompts import bias_prompt, summary_prompt, context_prompt
from archive.utils import chunk_article
from database import init_db, save_result
import wikipedia

# python -m spacy download en_core_web_sm

load_dotenv()
#openai.api_key = os.getenv("OPENAI_API_KEY")

def call_gpt(prompt, model="gpt-4"):
    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    response = client.responses.create(
        model="gpt-4o",
        instructions="You are a news analyst that talks like sherlock holmes providing clarity on biased news input",
        input=prompt,
    )
    return (response.output_text).strip()
    '''
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response['choices'][0]['message']['content'].strip()
    '''

def analyze_article(article):

    print("Running Summarization...")
    summary = call_gpt(summary_prompt(article))
    print("Summary:", summary, '\n')

    print(" Generating Historical Context...")
    context = call_gpt(context_prompt(article))
    print("Context:", context, '\n')

def get_context_from_wiki(query):
    try:
        summary = wikipedia.summary(query, sentences=5)
        return summary
    except wikipedia.DisambiguationError as e:
        return f"Disambiguation Error. Try: {e.options[:3]}"
    except Exception as e:
        return f"Error: {e}"

def main():
    print("Political Bias Classifier")
    print("------------------------")
    print("Enter article text to classify (type 'exit' to quit):")
    
    
    # Classify the text using ensemble model
    try:
        try:
            file = open("trial_article.txt", "r")
            user_input=file.read()
            file.close()
        except Exception as e:
            print(f"Error classifying text: {e}")

        bias, confidence = classify_article(user_input)
        print(f"\nPolitical bias: {bias.upper()} (confidence: {confidence:.2f})")
        
        # Also show individual model predictions if desired
        print("\nIndividual model predictions:")
        rf_bias, rf_conf = classify_article(user_input, 'rf')
        xgb_bias, xgb_conf = classify_article(user_input, 'xgb')
        nn_bias, nn_conf = classify_article(user_input, 'nn')
        
        print(f"Random Forest: {rf_bias.upper()} (confidence: {rf_conf:.2f})")
        print(f"XGBoost: {xgb_bias.upper()} (confidence: {xgb_conf:.2f})")
        print(f"Neural Network: {nn_bias.upper()} (confidence: {nn_conf:.2f})")

        analyze_article(user_input)
    except Exception as e:
        print(f"Error classifying text: {e}")

if __name__ == "__main__":
    main()