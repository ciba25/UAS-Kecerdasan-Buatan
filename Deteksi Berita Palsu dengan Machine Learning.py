#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# # CODE di atas merupakan import yang dibutuhkan

# In[14]:


#Read the data
df=pd.read_csv('Downloads/Programs/news.csv')

#Get shape and head
df.shape
df.head()


# # Code di atas merupakan code untuk memanggil data set yang sudah di ekstrak di direktori Programs.

# In[15]:


#DataFlair - Get the labels
labels=df.label
labels.head()


# # Code di atas untuk mendapatkan label dari data frame. Label yang dimaksud berupa keterangan "fake" atau "real"

# In[16]:


#DataFlair - Split the dataset
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)


# # code di atas berfungsi untuk membagi dan menguji kumpulan data

# In[17]:


#DataFlair - Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

#DataFlair - Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)


# # Paskan dan ubah vectorizer di kumpulan data

# In[18]:


#DataFlair - Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

#DataFlair - Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# # Selanjutnya, kita akan menginisialisasi PassiveAggressiveClassifier. Ini  akan memasang ini di tfidf_train dan y_train. Dari model ini, didapatkan akurasi sebesar 92.58%

# In[19]:


#DataFlair - Build confusion matrix
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])


# # print matriks kebingungan untuk mendapatkan wawasan tentang jumlah negatif dan positif palsu dan benar.

# In[ ]:




