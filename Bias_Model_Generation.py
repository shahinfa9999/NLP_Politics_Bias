import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier
from datasets import load_dataset

# Load and prepare data
csv_file = 'allsides_balanced_news_headlines-texts.csv'
data = pd.read_csv(csv_file)
data = data.dropna(subset=['text', 'bias_rating'])
cnn_daily_mail = load_dataset("cnn_dailymail", "3.0.0")
cnn_texts_train = cnn_daily_mail['train']['article']

xlsx_file = 'annotations.xlsx'
data_MBIC = pd.read_excel(xlsx_file)
data_MBIC = data_MBIC[['text', 'type']]
data_MBIC = data_MBIC.dropna(subset=['text', 'type'])
data_MBIC= data_MBIC.rename(columns={'type': 'bias_rating'})

data_MBIC_train = data_MBIC[:int(len(data_MBIC)*0.8)]
#data_MBIC_test = data_MBIC[int(len(data_MBIC)*0.8):]







# Function to determine the source more reliably
def determine_source(text):
    # Check for CNN mentions in a more reliable way
    if "(CNN)" in text[:200] or "CNN.com" in text or "CNN " in text[:200]:
        return "left"  # CNN is typically considered left-leaning
    # Check for Daily Mail mentions
    elif "Daily Mail" in text[:200] or "Rueters" in text[:200]:
        return "right"  # Daily Mail is typically considered right-leaning
    else:
        return None  # Skip articles we can't confidently classify

print("Processing CNN-DailyMail articles...")

# Create new dataframes for CNN-DailyMail articles
cnn_train_data = []
for article in cnn_daily_mail['train']['article']:
    source = determine_source(article)
    if source:
        cnn_train_data.append({
            'text': article,
            'bias_rating': source
        })



# Create new dataframes for CNN-DailyMail test articles
cnn_test_data = []
for article in cnn_daily_mail['test']['article']:
    source = determine_source(article)
    if source:
        cnn_test_data.append({
            'text': article,
            'bias_rating': source
        })

# Convert to dataframes
cnn_train_df = pd.DataFrame(cnn_train_data)
cnn_test_df = pd.DataFrame(cnn_test_data)

# Combine with original dataset
combined_data = pd.concat([data, cnn_train_df,data_MBIC_train,cnn_test_df], ignore_index=True)
#combined_data = combined_data[:10000]  # Limit to 10,000 rows for faster processing
combined_data = combined_data.dropna(subset=['text', 'bias_rating'])

print("balncing the dataset...")
# balance the dataset
# Separate the classes
left_data = combined_data[combined_data['bias_rating'] == 'left']
right_data = combined_data[combined_data['bias_rating'] == 'right']
n_samples=len(right_data)

combined_data = pd.concat([
    left_data.sample(n=n_samples, random_state=42),
    right_data.sample(n=n_samples, random_state=42)
], ignore_index=True)

#pre processing by removing CNN and DM from text to help pre processing

combined_data['text'] = (combined_data['text']
    .str.replace("(CNN)", "SOURCE", regex=False)
    .str.replace("CNN.com", "SOURCE", regex=False)
    .str.replace("CNN", "SOURCE", regex=False)
    .str.replace("Daily Mail", "SOURCE", regex=False)
    .str.replace("Rueters", "SOURCE", regex=False)
)

# more pre prcessing to lemmatize
import spacy
nlp = spacy.load('en_core_web_sm')

def preprocess_spacy(text):
    doc = nlp(text)
    lemmatized = [token.lemma_ for token in doc if not token.is_punct and not token.is_stop]
    return ' '.join(lemmatized)
print("leminatizing the text...")
combined_data['processed_text'] = combined_data['text'].apply(preprocess_spacy)

'''
for i in range(len(combined_data)):
    if "(CNN)" in combined_data['text'][i][:200] or "CNN.com" in combined_data['text'][i] or "CNN " in combined_data['text'][i][:200]:
        combined_data['text'][i] = combined_data['text'][i].replace("(CNN)", "")
        combined_data['text'][i] = combined_data['text'][i].replace("(CNN)", "")
        combined_data['text'][i] = combined_data['text'][i].replace("CNN.com", "")
        combined_data['text'][i] = combined_data['text'][i].replace("CNN", "")
        continue
    if "Daily Mail" in combined_data['text'][i][:200] or "Rueters" in combined_data['text'][i][:200]:
        combined_data['text'][i] = combined_data['text'][i].replace("Daily Mail", "")
        combined_data['text'][i] = combined_data['text'][i].replace("Rueters", "")
        continue
'''
# Create label mappings
#label_mapping = {'left': 0, 'center': 1, 'right': 2}
label_mapping = {'left': 0, 'right': 1}

# Filter data to only include rows with valid labels
valid_labels = list(label_mapping.keys())
combined_data = combined_data[combined_data['bias_rating'].isin(valid_labels)]

# Extract cleaned data
texts = combined_data['processed_text'].tolist()
#texts = combined_data['text'].tolist()
labels = [label_mapping[label] for label in combined_data['bias_rating'].tolist()]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# Create document-term matrix
print("Creating document-term matrix...")
vectorizer = CountVectorizer(max_features=5000, stop_words='english')
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Train LDA model
print("Training LDA model...")
n_topics = 15 #20  # tune this hyperparameter
lda = LatentDirichletAllocation(
    n_components=n_topics,
    max_iter=20,
    learning_method='online',
    learning_decay=0.7,  # Adjust this value
    random_state=42,
    batch_size=128,
    n_jobs=-1
)
X_train_topics = lda.fit_transform(X_train_counts)
X_test_topics = lda.transform(X_test_counts)



# Train a classifier on LDA features
print("Training classifier on LDA features...")
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train_topics, y_train)

# Make predictions
y_pred = clf.predict(X_test_topics)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

# Print results
print(f"\nResults using LDA with {n_topics} topics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=valid_labels))
'''
# For LightGBM
import lightgbm as lgb

# Install with: pip install lightgbm
print("Training LightGBM classifier...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=7,
    objective='multiclass',
    class_weight='balanced',
    random_state=42
)
lgb_model.fit(X_train_topics, y_train)
y_pred_lgb = lgb_model.predict(X_test_topics)

# Evaluate
print("\nResults using LightGBM on combined features:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lgb):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_lgb, average='weighted'):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lgb, target_names=valid_labels))
'''


# Example function for classifying new texts
def classify_political_bias(text, vectorizer, lda, classifier, id2label):
    # Convert text to document-term matrix
    X_counts = vectorizer.transform([text])
    # Transform to topic distribution
    X_topics = lda.transform(X_counts)
    # Predict label
    pred = classifier.predict(X_topics)[0]
    # Convert numeric label to text
    return id2label[pred]

# Test on a few examples
test_texts = [
    "The president's tax plan will significantly benefit corporations while hurting middle-class Americans.",
    "Both parties have shown willingness to compromise on the infrastructure bill.",
    "Traditional family values must be protected against the radical agenda."
]

#id2label = {0: 'left', 1: 'center', 2: 'right'}
id2label = {0: 'left', 1: 'right'}



import torch
from transformers import AutoTokenizer, AutoModel

# Load BERT model
#tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#model = AutoModel.from_pretrained("bert-base-uncased")

tokenizer = AutoTokenizer.from_pretrained("roberta-base")  # or try "nlptown/bert-base-multilingual-uncased-sentiment"
model = AutoModel.from_pretrained("roberta-base")

# Function to get BERT embeddings
def get_bert_embeddings(texts, batch_size=16):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        # Use CLS token embedding
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

# Get embeddings
print("Generating BERT embeddings for training set...")
X_train_bert = get_bert_embeddings(X_train, batch_size=4)
print("Generating BERT embeddings for test set...")
X_test_bert = get_bert_embeddings(X_test, batch_size=4)

# Combine with LDA topics
X_train_combined = np.hstack([X_train_topics, X_train_bert])
X_test_combined = np.hstack([X_test_topics, X_test_bert])

# Train classifier on combined features
print("Training classifier on combined features...")
from sklearn.ensemble import RandomForestClassifier
clf_combined = RandomForestClassifier(
    n_estimators=200, 
    random_state=42,
    class_weight='balanced'
)
clf_combined.fit(X_train_combined, y_train)

# Make predictions
y_pred_combined = clf_combined.predict(X_test_combined)

# Calculate metrics
accuracy_combined = accuracy_score(y_test, y_pred_combined)
f1_combined = f1_score(y_test, y_pred_combined, average='weighted')

# Print results
print("\nResults using Combined LDA+BERT features:")
print(f"Accuracy: {accuracy_combined:.4f}")
print(f"F1 Score: {f1_combined:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_combined, target_names=valid_labels))


# For XGBoost
import xgboost as xgb

# Install with: pip install xgboost
print("Training XGBoost classifier...")
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.01,
    max_depth=7,
    objective='binary:logistic', #objective='binary:logistic', objective='multi:softprob'
    num_class=1, #3
    eval_metric='logloss', # 'mlogloss'
    random_state=42,
    use_label_encoder=False
)
xgb_model.fit(X_train_combined, y_train)
y_pred_xgb = xgb_model.predict(X_test_combined)
print("XGBoost predictions shape:", y_pred_xgb.shape)
print("First 5 predictions:", y_pred_xgb[:5])
# Evaluate
print("\nResults using XGBoost on combined features:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_xgb, average='weighted'):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb, target_names=valid_labels)) 

# NNN
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

# Scale combined features
print("Scaling combined features at NNN stage...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_combined)
X_test_scaled = scaler.transform(X_test_combined)
y_train = np.array(y_train)

# Build the neural network
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))  # 2-class classification


# Compile the model
#loss='sparse_categorical_crossentropy'
#loss = 'binary_crossentropy'
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Model compiled.")

# Train the model
epochs = 5
batch_size =128

model.fit(X_train_scaled , y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)
print("Modeled trained.")

# Evaluate the model on the test data
y_test = np.array(y_test)
loss, accuracy = model.evaluate(X_test_scaled, y_test)
ypred = model.predict(X_test_scaled)
y_pred_proc = np.where(ypred > 0.5, 1, 0)  # Convert probabilities to binary predictions
#y = np.argmax(ypred, axis=1)  # For multi-class classification
print("\nResults using NNN on combined features:")
print(f"Accuracy: {accuracy:.4f}")
print(f'Loss: {loss:.4f}')
print(f"F1 Score: {f1_score(y_test, y_pred_proc, average='weighted'):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_proc, target_names=valid_labels)) 




'''
print("\nResults using NNN on combined features:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_xgb, average='weighted'):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb, target_names=valid_labels))
'''

'''


# Get BERT embeddings
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Function to get BERT embeddings
def get_bert_embeddings(texts, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        # Use CLS token embedding or mean of all token embeddings
        batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

# Combine LDA topics with BERT embeddings
X_train_bert = get_bert_embeddings(X_train)
X_test_bert = get_bert_embeddings(X_test)

X_train_combined = np.hstack([X_train_topics, X_train_bert])
X_test_combined = np.hstack([X_test_topics, X_test_bert])

# Train classifier on combined features
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train_combined, y_train)
'''



import joblib
import numpy as np

# Save Random Forest
joblib.dump(clf_combined, 'random_forest_model.pkl')

# Save XGBoost
xgb_model.save_model('xgboost_model.json')

# Save Neural Network
model.save('neural_network_model.h5')

# Save LDA
joblib.dump(lda, 'lda_model.pkl')

# Save Vectorizer
joblib.dump(vectorizer, 'vectorizer.pkl')

# Save Combined Features
np.save('X_train_combined.npy', X_train_combined)
np.save('X_test_combined.npy', X_test_combined)

print("All models and features saved.")