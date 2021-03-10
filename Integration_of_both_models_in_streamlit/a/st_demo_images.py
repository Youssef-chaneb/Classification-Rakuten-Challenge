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
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.applications.resnet50 import ResNet50
from keras import backend as K
from keras.callbacks import Callback
from keras.layers import Dense
from keras.models import Model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.sequence import pad_sequences
import cv2 
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

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
    st.title('Rakuten image Classification application')
    st.info('Classification based on the image of the product')
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg','png','jpeg'])
    if uploaded_file is not None:
        image_data = Image.open(uploaded_file)
        st.image(image_data, caption='Uploaded Image.', use_column_width=True)
        img = ImageOps.fit(image_data,(224,224))
    base_model = ResNet50(weights='imagenet', include_top=False) 
    for layer in base_model.layers: 
        layer.trainable = False

    model = Sequential()
    model.add(base_model) # Ajout du modèle ResNet50
    model.add(GlobalAveragePooling2D()) 
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(rate=0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(rate=0.2))
    model.add(Dense(27, activation='softmax')) 
    model.load_weights('resnet.h5')
    
    inputs = np.array(img).reshape((1, 224, 224, 3)).astype('float32')
    prediction=np.argmax(model.predict(inputs))
    prediction=num_cat(prediction)
    nom_catégorie=name_cat(prediction)
    st.write(prediction,'(',nom_catégorie,')')

if __name__ == '__main__':
    main()
# %%
# %%


# %%
