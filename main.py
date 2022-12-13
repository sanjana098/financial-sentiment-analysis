import pickle
import string

import streamlit as st 
import pandas as pd 
import numpy as np 

import tensorflow as tf
from keras.utils import pad_sequences

import ssl
import nltk

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer 


categories = ["Negative", "Neutral", "Positive"]


def preprocess(x):
    # Remove lowercase, punctuations, lemmatize
    
    lemmatizer = WordNetLemmatizer()
    
    text_lcase = x.lower()
    text_no_punct = text_lcase.translate(str.maketrans('', '', string.punctuation))
    
    return ' '.join(lemmatizer.lemmatize(t) for t in text_no_punct.split())

st.title('Financial sentiment analysis')
title = st.text_input('Movie title', 'Life of Brian')

st.write(title)

cleaned_title = preprocess(title)

model = tf.keras.models.load_model('model.keras')

tokenizer = None
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

tokenized_input = tokenizer.texts_to_sequences([cleaned_title])
tokenized_input = pad_sequences(tokenized_input, maxlen=52, padding='post')
# print(tokenized_input)

res = model.predict(tokenized_input)

cat = categories[np.argmax(res)]
st.write(cat)