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


categories = ["Negative", "Neutral", "Positive"]

def preprocess(x):
    maxlen = 52
    lemmatizer = WordNetLemmatizer()
    
    text_lcase = x.lower()
    text_no_punct = text_lcase.translate(str.maketrans('', '', string.punctuation))
    
    return ' '.join(lemmatizer.lemmatize(t) for t in text_no_punct.split()[:maxlen])

st.title('Financial sentiment analysis')
title = st.text_input('Enter a financial news headline', 'Tesla shares have fallen 28% since Elon Musk took over Twitter, lagging other carmakers')

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

category = np.argmax(res)

if category == 0:
    st.error(title, icon="😣 Negative: ")
elif category == 1:
    st.warning(title, icon="😐 Neutral: ")
else:
    st.success(title, icon="😊 Positive: ")

# st.error('This is an error', icon="🚨")
# st.warning('This is a warning', icon="😐")
# st.info('This is a purely informational message', icon="ℹ️")