# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 12:49:08 2022

@author: Rohit Rankhamb
"""

import streamlit as st
import pickle
import nltk
from nltk.stem.porter import PorterStemmer 
from nltk.corpus import stopwords
import string

ps = PorterStemmer()
tfidf = pickle.load(open("D:/SMS Spam Classifier/vectorizer.pkl",'rb'))
model = pickle.load(open("D:/SMS Spam Classifier/model.pkl",'rb'))

# lower case
def transform_text(text):
    text=text.lower()         # Convert all words in lower case   
# Tokenization
    text=nltk.word_tokenize(text)   # Split our sentence in word
# Removing special characters
    y=[]             # Create a empty list
    for i in text:
        if i.isalnum():
            y.append(i)
# Removing stop words and punctuation
    text=y[:]      
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)                       
# Stemming
    text=y[:]
    y.clear()    
    for i in text:
        y.append(ps.stem(i))            
    return " ".join(y)

# Title
st.title('Email/SMS Spam Classifier')

# Input
input_sms = st.text_input('Enter your SMS/Email') 

# Creating button for analysis
if st.button('Analyse'):

    # Preprocessing text
    transformed_sms = transform_text(input_sms)
    
    # Vextorize
    vector_input = tfidf.transform([transformed_sms])
    
    # Analyse
    result = model.predict(vector_input)[0]
    
    # Display
    if result == 1:
        st.header('This is a Spam')
    else:
        st.header('This is Not Spam')