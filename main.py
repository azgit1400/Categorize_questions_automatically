# text preprocessing modules
from string import punctuation
# text preprocessing modules
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re  # regular expression
import os
from os.path import dirname, join, realpath
import joblib
import uvicorn
import tensorflow_hub as hub
from fastapi import FastAPI
import pandas as pd

# import necessary libraries
import tensorflow_hub as hub

app = FastAPI(
    title="Question Model API",
    description="A simple API that use NLP model to predict the tag of questions",
    version="0.1",
)

# load the model
multilabel_binarizer = joblib.load("multilabel_binarizer.pkl", 'r')
model = joblib.load("rfc_final_model.pkl", 'r')
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    
# cleaning the data
def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    # Clean the text, with the option to remove stop_words and to lemmatize word
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"http\S+", " link ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\s+", "", text)  # remove numbers
    
    # Remove punctuation from text
    text = "".join([c for c in text if c not in punctuation])
    
    # Optionally, remove stop words
    if remove_stop_words:
        # load stopwords
        stop_words = stopwords.words("english")
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
        
    # Optionally, shorten words to their stems
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)
        
    # Return a list of words
    return text

@app.get("/predict-tag")
def predict_sentiment(question: str):
    """
    A simple function that receive a question content and predict the tag of the content.
    :param quesion:
    :return: prediction, probabilities
    """
    # clean the question
    cleaned_question = text_cleaning(question)
    
    # Embedding with USE.
    X = embed([cleaned_question]).numpy()
    
    # perform prediction
    predict = model.predict(X)
    predict_probas = model.predict_proba(X)
    # Inverse multilabel binarizer
    tags_predict = multilabel_binarizer.inverse_transform(predict)
    
    # DataFrame of probas
    df_predict_probas = pd.DataFrame(columns=['Tags', 'Probas'])
    df_predict_probas['Tags'] = multilabel_binarizer.classes_
    df_predict_probas['Probas'] = predict_probas.reshape(-1)
    # Select probas > 33%
    df_predict_probas = df_predict_probas[df_predict_probas['Probas']>=0.33]\
        .sort_values('Probas', ascending=False)

    # Results
    results = {}
    results['Predicted_Tags'] = tags_predict
    results['Predicted_Tags_Probabilities'] = df_predict_probas\
        .set_index('Tags')['Probas'].to_dict()
    return result
