import string

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
#reading training data
df=pd.read_csv('train.txt', sep=';',header=None,names =['text', 'emotion'])

#converting all text to lower case
df['text']= df['text'].apply(lambda x : x.lower())
#removing punctuation from text
def remove_punc(txt):
  if isinstance(txt,str):
     return txt.translate(str.maketrans('','',string.punctuation))
  return txt
df['text']=df['text'].apply(remove_punc)

#removing digits

def remove_no(txt):
  if not isinstance(txt, str):
    return ""
  new = ""
  for char in txt:
    if not char.isdigit():
      new += char
  return new
df['text']=df['text'].apply(remove_no)

#removing emojis
def remove_emojis(txt):
  new=""
  for  i in txt:
    if i.isascii():
      new+=i
  return new

df['text']=df['text'].apply(remove_emojis)


# Now we will remove  is , was , for etc type of helping words
# or called stopwords

# because it creates noice
# to remove them we can use NLTK or spacy libraries
import nltk
import ssl
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))
def remove(txt):
  words= txt.split()
  cleaned=[]
  for i in words:
    if not i in stop_words :
      cleaned.append(i)

  return ' '.join(cleaned)
df['text']= df['text'].apply(remove)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['emotion'], test_size=0.2, random_state=42)

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# bow_vectorizer = CountVectorizer()
# x_train_bow= bow_vectorizer.fit_transform(X_train)
# x_test_bow = bow_vectorizer.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# nb_model=MultinomialNB()
# nb_model.fit(x_train_bow, y_train)

# pred_nb =nb_model.predict(x_test_bow)

# print("Accuracy:", accuracy_score(y_test, pred_nb))

tfidf_vectorizer =TfidfVectorizer()
x_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
x_test_tfidf = tfidf_vectorizer.transform(X_test)

# nb2_model= MultinomialNB()
# nb2_model.fit(x_train_tfidf, y_train)
# pred_nb2 =nb2_model.predict(x_test_tfidf)

# print("Accuracy:", accuracy_score(y_test, pred_nb2))

from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression(max_iter=1000)
logistic_model = logistic_model.fit(x_train_tfidf, y_train)
log_pred = logistic_model.predict(x_test_tfidf)
print("Accuracy:", accuracy_score(y_test, log_pred))