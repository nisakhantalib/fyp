from flask import Flask, request, render_template
import joblib
import torch
import numpy as np
from transformers import RobertaTokenizer, DistilBertTokenizer, RobertaForSequenceClassification, DistilBertForSequenceClassification, DistilBertModel
from sklearn.preprocessing import LabelEncoder
from scipy.special import softmax
from scipy.stats import mode
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
from torch.utils.data import DataLoader, Dataset
import textstat
from collections import Counter
from nltk import word_tokenize, pos_tag, sent_tokenize
from textstat import flesch_reading_ease, flesch_kincaid_grade, gunning_fog, dale_chall_readability_score

from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import nltk
from nltk.corpus import stopwords
import tensorflow as tf
import torch.nn as nn


app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load saved models and components
svm_pipeline=joblib.load('models/svm_tfidf.pkl')
tfidf_vectorizer_svm=joblib.load('models/tfidf_vectorizer_svm.pkl')

xgboost_pipeline = joblib.load('models/xgb_tfidf.pkl')
tfidf_vectorizer_xgb = joblib.load('models/tfidf_vectorizer_xgb.pkl')

logreg_pipeline = joblib.load('models/logreg_tfidf.pkl')
tfidf_vectorizer_logreg = joblib.load('models/tfidf_vectorizer_logreg.pkl')

xgbstylo_pipeline = joblib.load('models/xgb_tfidf_stylo.pkl')
tfidf_vectorizer_xgbstylo = joblib.load('models/tfidf_vectorizer_xgbstylo.pkl')

tfidf_softvoting_ml = joblib.load('models/voting_classifier_xgb_rf_lr.pkl')
logistic_regression_model = joblib.load('models/logistic_regression_fordistilbert.joblib')

label_encoder = joblib.load('models/label_encoder.pkl')



tfidf_vectorizer_softvoting_ml = joblib.load('models/tfidf_vectorizer_sv_ml.pkl')

num_labels = 51  
roberta_model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=num_labels)
roberta_model.load_state_dict(torch.load('models/roberta_finetuned_weight_5epochs.pth', map_location=device))
roberta_model.to(device)
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

distilbert_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)
distilbert_model.load_state_dict(torch.load('models/distilbert_finetuned_weight_5epochs.pth', map_location=device))
distilbert_model.to(device)
distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')



class DNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DNNClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x



input_dim, hidden_dim, output_dim=5000,512, 51
tfidf_vectorizer_dnn = joblib.load('models/tfidf_vectorizer_dnn.pkl')
dnn_model = DNNClassifier(input_dim, hidden_dim, output_dim)  
dnn_model.load_state_dict(torch.load('models/dnn_classifier.pth', map_location=device))
dnn_model.to(device)


svm_classifier = joblib.load('models/svm_distilbert_stylo2_attention_model.pkl')
scaler = joblib.load('models/scaler_distilbert_stylo2_attention.pkl')

# Load data from Excel file once at the start of the app
data = pd.read_excel('data/50human15eachgpt4withnumber.xlsx')


# Function to extract stylometric features
def extract_stylometric_features(text, max_length=250):
    tokens = word_tokenize(text.lower())
    sentences = sent_tokenize(text)
    num_characters = len(text)
    num_words = len(tokens)
    num_sentences = len(sentences)
    num_unique_words = len(set(tokens))
    lexical_diversity = num_unique_words / num_words if num_words else 0
    avg_word_length = np.mean([len(word) for word in tokens]) if tokens else 0
    avg_sentence_length = np.mean([len(sentence.split()) for sentence in sentences]) if sentences else 0
    function_words_count = sum(token in stopwords.words('english') for token in tokens)
    hapax_legomena = len([word for word in set(tokens) if tokens.count(word) == 1])
    hapax_dislegomena = len([word for word in set(tokens) if tokens.count(word) == 2])
    avg_syllables_per_word = np.mean([textstat.syllable_count(word) for word in tokens]) if tokens else 0
    fog_index = textstat.gunning_fog(text)
    smog_index = textstat.smog_index(text)
    readability_index = textstat.flesch_reading_ease(text)
    # Combine all features into a single list
    features = [num_characters, num_words, num_sentences, num_unique_words, lexical_diversity, avg_word_length, avg_sentence_length, function_words_count, hapax_legomena, hapax_dislegomena, avg_syllables_per_word, fog_index, smog_index, readability_index]
    return features[:max_length]  # Truncate or pad the feature vector to max_length


# other set of stylometric features
def extract_stylometric_features2(text, max_features=50):  
    tokens = word_tokenize(text.lower())
    sentences = sent_tokenize(text)
    num_sentences = len(sentences)
    num_words = len(tokens)
    num_unique_words = len(set(tokens))
    num_characters = len(text)
    lexical_diversity = num_unique_words / num_words if num_words else 0
    avg_word_length = np.mean([len(word) for word in tokens]) if tokens else 0
    avg_sentence_length = num_words / num_sentences if num_sentences else 0
    tagged_tokens = pos_tag(tokens)
    pos_counts = Counter(tag for word, tag in tagged_tokens)
    normalized_pos_counts = [pos_counts[tag] / num_words if num_words else 0 for tag, _ in Counter(tagged_tokens).most_common(max_features)]
    while len(normalized_pos_counts) < max_features:
        normalized_pos_counts.append(0)  # Ensure all vectors have the same length
    features = [
        num_words, num_unique_words, num_characters, lexical_diversity,
        avg_word_length, avg_sentence_length, num_sentences,
        flesch_reading_ease(text), flesch_kincaid_grade(text),
        gunning_fog(text), dale_chall_readability_score(text)
    ] + normalized_pos_counts[:max_features]  # Ensure the feature vector is of fixed length
    return features

def extract_stylometric_features3(text):
    tokens = word_tokenize(text.lower())
    sentences = sent_tokenize(text)
    num_sentences = len(sentences)
    num_words = len(tokens)
    num_unique_words = len(set(tokens))
    num_characters = len(text)
    lexical_diversity = num_unique_words / num_words if num_words else 0
    avg_word_length = np.mean([len(word) for word in tokens]) if tokens else 0
    avg_sentence_length = num_words / num_sentences if num_sentences else 0
    tagged_tokens = pos_tag(tokens)
    pos_counts = Counter(tag for word, tag in tagged_tokens)
    total_pos_counts = sum(pos_counts.values())
    noun_ratio = pos_counts['NN'] / total_pos_counts if 'NN' in pos_counts else 0
    adjective_ratio = pos_counts['JJ'] / total_pos_counts if 'JJ' in pos_counts else 0
    verb_ratio = pos_counts['VB'] / total_pos_counts if 'VB' in pos_counts else 0
    flesch_reading_ease = textstat.flesch_reading_ease(text)
    return np.array([lexical_diversity, avg_word_length, avg_sentence_length, noun_ratio, adjective_ratio, verb_ratio, flesch_reading_ease])


def get_distilbert_embedding(text):
    inputs = distilbert_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        # Ensure to request hidden states
        outputs = distilbert_model(**inputs, output_hidden_states=True)
        # Accessing the last hidden state
        last_hidden_state = outputs.hidden_states[-1]
        # Apply mean pooling or any required operation
        return torch.mean(last_hidden_state, dim=1).cpu().numpy()

def get_combined_features(text):
    distilbert_embedding = get_distilbert_embedding(text).flatten()
    stylometric_features = extract_stylometric_features(text)
    combined_features = np.hstack([distilbert_embedding, stylometric_features])
    return combined_features.reshape(1, -1)

#for embed_sv
def extract_single_feature(model, text, tokenizer, device):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model(**inputs)
        feature = outputs.last_hidden_state[:,0,:].cpu().numpy()  # Extract the embeddings of the [CLS] token
    return feature.flatten()





def predict_author(text, model_choice):
    if model_choice == 'svm':
        # SVM predictions (already in probability format)
        text_transformed = tfidf_vectorizer_svm.transform([text])
        final_probs = svm_pipeline.predict_proba(text_transformed)
    elif model_choice == 'xgboost':
        # XGBoost predictions (already in probability format)
        text_transformed = tfidf_vectorizer_xgb.transform([text])
        final_probs = xgboost_pipeline.predict_proba(text_transformed)

    elif model_choice == 'logreg':
        # logreg predictions (already in probability format)
        text_transformed = tfidf_vectorizer_logreg.transform([text])
        final_probs = logreg_pipeline.predict_proba(text_transformed)

    elif model_choice == 'xgboost_stylo':
        # logreg predictions (already in probability format)
        text_transformed = tfidf_vectorizer_xgbstylo.transform([text])
        # Extract the same stylometric features used during training
        stylo_features = extract_stylometric_features(text)
        stylo_features = np.array(stylo_features).reshape(1, -1)
        
        # Combine the TF-IDF features and stylometric features for the new text
        tfidf_stylo = np.hstack((text_transformed.toarray(), stylo_features))
        
        # Predict probabilities with the combined XGBoost model
        final_probs = xgbstylo_pipeline.predict_proba(tfidf_stylo)


    elif model_choice == 'roberta':
        # RoBERTa predictions
        final_probs = get_model_probabilities(text, roberta_model, roberta_tokenizer)

    elif model_choice == 'distilbert':
        # DistilBERT predictions
        final_probs = get_model_probabilities(text, distilbert_model, distilbert_tokenizer)
    elif model_choice == 'softvoting_ml':  
        text_transformed = tfidf_vectorizer_softvoting_ml.transform([text])
        final_probs = tfidf_softvoting_ml.predict_proba(text_transformed)


    elif model_choice == 'softvoting_llm':
        # Soft voting
        xgb_probs = xgboost_pipeline.predict_proba(tfidf_vectorizer_xgb.transform([text]))
        roberta_probs = get_model_probabilities(text, roberta_model, roberta_tokenizer)
        distilbert_probs = get_model_probabilities(text, distilbert_model, distilbert_tokenizer)
        final_probs = np.mean([xgb_probs, roberta_probs, distilbert_probs], axis=0)
    elif model_choice == 'hardvoting':
        #Hard voting
        xgb_pred= xgboost_pipeline.predict(tfidf_vectorizer_xgb.transform([text]))[0] #predicted class, not probabilities
        roberta_pred = get_model_class_prediction(text, roberta_model, roberta_tokenizer)
        distilbert_pred = get_model_class_prediction(text, distilbert_model, distilbert_tokenizer)
        predictions=[xgb_pred, roberta_pred, distilbert_pred]
        most_common= Counter(predictions).most_common(1)
        final_pred=most_common[0][0]
        return label_encoder.inverse_transform([final_pred])[0]
    
    elif model_choice == 'tfidf_dnn': #model41
        text_transformed = tfidf_vectorizer_dnn.transform([text]).toarray()  # Transform text to TF-IDF features
        text_tensor = torch.tensor(text_transformed, dtype=torch.float).to(device)  # Convert to tensor and move to the correct device
        dnn_model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Turn off gradients for validation
            outputs = dnn_model(text_tensor)  # Get model outputs for the transformed text
            _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
            final_prediction = label_encoder.inverse_transform([predicted.item()])[0]  # Decode the predicted label
        return final_prediction  # Return the predicted label

    elif model_choice == 'distilbert_stylo_attention_svm': #model36, early fusion
        text_features = get_distilbert_embedding(text).flatten()
        stylometric_features =np.array( extract_stylometric_features3(text))
        # Combine features and reshape for the scaler
        combined_features = np.hstack([text_features, stylometric_features])
        combined_features = combined_features.reshape(1, -1)  # Ensure it is a 2D array
        combined_features_scaled = scaler.transform(combined_features)  # Now, this should match the expected size

        svm_prediction = svm_classifier.predict(combined_features_scaled)
        final_prediction = label_encoder.inverse_transform([svm_prediction[0]])[0]  # Decode the prediction into label
        return final_prediction






    elif model_choice == 'distilbert_logreg': #model31, stacking
        # Preprocess the text and generate embeddings using DistilBERT
        inputs = distilbert_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            model_outputs = distilbert_model(**inputs)
            distilbert_features = model_outputs.logits.cpu().numpy()  # Extract logits as features

        # Predict with Logistic Regression model using DistilBERT outputs as input
        final_predictions = logistic_regression_model.predict_proba(distilbert_features)
        final_prediction = np.argmax(final_predictions, axis=1)[0]
        return label_encoder.inverse_transform([final_prediction])[0]



    else:
        final_probs = None  # Default case if no valid option is selected

    final_prediction = np.argmax(final_probs, axis=1)[0] if final_probs is not None else None
    return label_encoder.inverse_transform([final_prediction])[0] if final_prediction is not None else "Unknown"

# Helper function for transformer models to get probability predictions
def get_model_probabilities(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = softmax(outputs.logits.detach().cpu().numpy(), axis=1)
    return probs

def get_model_class_prediction(text, model, tokenizer):
    # Helper function to predict class (not probabilities) with transformer models
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]  # Get class prediction
    return prediction


@app.route('/', methods=['GET', 'POST'])
def index():
    texts_list = data.to_dict(orient='records')
    selected_text_index = None
    model_selected = None
    predicted_author = None
    true_author = None

    if request.method == 'POST':
        selected_text_index = request.form.get('text_selection')
        model_selected = request.form.get('model_choice')
        if selected_text_index.isdigit():
            index = int(selected_text_index)
            if 0 <= index < len(texts_list):
                selected_item = texts_list[index]
                selected_text = selected_item['text']
                true_author = selected_item['label']
                predicted_author= predict_author(selected_text, model_selected)
    
    return render_template('index.html',
                            texts=texts_list,
                            selected_text_index=selected_text_index,
                            model_selected=model_selected,
                            predicted_author=predicted_author,
                            true_author=true_author)



if __name__ == '__main__':
    app.run(debug=True)







