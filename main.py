import pickle
import string

import streamlit as st 
import pandas as pd 
import numpy as np 

import tensorflow as tf
from keras.utils import pad_sequences

import nltk

nltk.download('omw-1.4')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer 

# Load the saved Keras LSTM model
model = tf.keras.models.load_model('model.keras')

# Load the tokenizer learned from original dataset
tokenizer = None
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

def preprocess(x):
    maxlen = 52
    lemmatizer = WordNetLemmatizer()
    
    text_lcase = x.lower()
    text_no_punct = text_lcase.translate(str.maketrans('', '', string.punctuation))
    
    return ' '.join(lemmatizer.lemmatize(t) for t in text_no_punct.split()[:maxlen])

st.title('Financial sentiment analysis')
text = st.text_input('Enter a financial news headline', 'Tesla shares have fallen 28% since Elon Musk took over Twitter, lagging other carmakers')

cleaned_text = preprocess(text)

tokenized_input = tokenizer.texts_to_sequences([cleaned_text])
tokenized_input = pad_sequences(tokenized_input, maxlen=52, padding='post')

res = model.predict(tokenized_input)
category = np.argmax(res)

if category == 0:
    st.error( 'Negative: ' + text, icon="😣")
elif category == 1:
    st.warning('Neutral: ' + text, icon="😐")
else:
    st.success('Positive: ' + text, icon="😊")
