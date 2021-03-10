#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import numpy as np 
import re 
import string
import pickle
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

with open('tokenizer.pickle', 'rb') as handle:
    news_vec = pickle.load(handle)

target=pd.read_csv('Y_train_CVw08PX.csv')


def num_cat(x):
    a=np.arange(0,27)
    b=sorted(target.prdtypecode.unique())
    for i in range(len(a)):
        if x==a[i]:
            return b[i]

def name_cat(x):
    a=sorted(target.prdtypecode.unique())
    b=['Livres','Jeux vidéos',
    'Matériel gaming','Consoles de jeux','Figurines/accessoires','Cartes de collection','Figurines/accessoire 2',
    'Jouets enfants','Jeux de société','Jouets avions/voitures','Jeux de société','Figurines/jeux plein air',
    'Puériculture','Meuble, accessoires et literie Maison','Linge de maison','Alimentation','Décoration',
    'Animalerie','Journal et Magazine','Livres d apprentissages ','Consoles de jeux 2','Papeterie','Jardin et piscine 1',
    'Jardin et piscine 2','Jardin et piscine 3','Livres 2','Jeux vidéos 2']
    for i in range(len(a)):
        if x==a[i]:
            return b[i]
# %%
def main():
    '''clf app with streamlit'''
    st.title('Rakuten Text Classification application')
    st.info('Classification based on the title and the description of the product')

    Title = st.text_area('Enter Title')
    Description = st.text_area('Enter Description')
    news_text = Title + Description   
    if st.button('Classify'):
        st.text('Original test ::\n{}'.format(news_text))
        text=[news_text]
        vect_text=news_vec.texts_to_sequences(text)
        vect_text=pad_sequences(vect_text, maxlen=1500, padding="post", truncating="post")
        predictor= keras.models.load_model("cnn_conv1D")
        prediction=np.argmax(predictor.predict(vect_text))
        prediction=num_cat(prediction)
        nom_catégorie=name_cat(prediction)
        st.write(prediction,'(',nom_catégorie,')')
if __name__ == '__main__':
    main()
# %%
# %%

