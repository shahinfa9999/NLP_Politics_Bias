import os
import openai
from dotenv import load_dotenv
from prompts import bias_prompt, summary_prompt, context_prompt
from archive.utils import chunk_article
from database import init_db, save_result
import wikipedia

# python -m spacy download en_core_web_sm

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def call_gpt(prompt, model="gpt-4"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response['choices'][0]['message']['content'].strip()

def analyze_article(file_path):
    with open(file_path, 'r') as f:
        article = f.read()

    '''
    print(" Running Bias Detection...")
    bias = call_gpt(bias_prompt(article))
    print("🧭 Bias:", bias, '\n')
    '''

    print("Running Summarization...")
    summary = call_gpt(summary_prompt(article))
    print("Summary:", summary, '\n')

    print(" Generating Historical Context...")
    context = call_gpt(context_prompt(article))
    print("📚 Context:", context, '\n')

def get_context_from_wiki(query):
    try:
        summary = wikipedia.summary(query, sentences=5)
        return summary
    except wikipedia.DisambiguationError as e:
        return f"Disambiguation Error. Try: {e.options[:3]}"
    except Exception as e:
        return f"Error: {e}"
    
import joblib
import numpy as np
import re
from tensorflow.keras.models import load_model

# Load everything
model = load_model('bias_classifier_model.h5')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
lda_model = joblib.load('lda_model.pkl')
scaler = joblib.load('scaler.pkl')

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Predict function
def predict_article_bias(text):
    cleaned = clean_text(text)
    tfidf_vec = tfidf_vectorizer.transform([cleaned])
    lda_feat = lda_model.transform(tfidf_vec)
    combined = np.hstack([tfidf_vec.toarray(), lda_feat])
    scaled = scaler.transform(combined)
    
    pred = model.predict(scaled)
    label = 'right' if pred[0][0] > 0.5 else 'left'
    print(f'Predicted Bias: {label}')
    return label
# Example
if __name__ == "__main__":
    article = "President's tax reform is receiving criticism from conservative circles."
    predict_article_bias(article)

if __name__ == "__main__":
    analyze_article("articles/sample_article.txt")
    topic = call_gpt(f"What is the main topic of this article?\n{article}")
    wiki_context = get_context_from_wiki(topic)
    print("📚 Wiki Context:", wiki_context)
    conn = init_db()
    # After analysis:
    save_result(conn, file_path, bias, summary, context, wiki_context)
