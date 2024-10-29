# Kütüphaneler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# Veri yükleme
fake_news = pd.read_csv('fake_news.csv')
true_news = pd.read_csv('true_news.csv')

# "Veri yüklendi mi?" testi
#print(fake_news.head())
#print(true_news.head())

# Sahte ile gerçek haberler için 1 ve 0 etiketlerinin girilmesi
fake_news['label'] = 0
true_news['label'] = 1

#print(fake_news.head())
#print(true_news.head())

# Sahte ve gerçek haberlerin birleştirilmesi
news = pd.concat([fake_news,true_news], axis=0)

#print(news.head())
#print(news.tail())

# Boş veri olup olmadığının kontrolü
#print(news.isnull().sum())

# Sahte ve gerçek haberlerin sırayla gelmemesi için karıştırılması
news = news.sample(frac=1)
#print(news.head())

# Karıştırıldıktan sonra oluşan index sayılarını düzenleme
news.reset_index(inplace=True)
#print(news.head())

news.drop(['index'], axis=1, inplace=True)
print(news.head(10))


# Verilerin Ön İşlemesi
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
print(stop_words)
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

def word_operations(text):
    # Küçük harfe dönüştürülmesi
    text = text.lower()
    
    # Linklerin silinmesi
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # HTML Taglerinin silinmesi
    text = re.sub(r'<.*?>', ' ', text)
    
    # Noktalama işaretlerinin silinmesi
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Sayıların silinmesi
    text = re.sub(r'\d', ' ', text)
    
    # Yeni satır karakterlerinin silinmesi (\n)
    text = re.sub(r'\n', ' ', text)
    
    # Sekme karakterlerinin silinmesi (\t)
    text = re.sub(r'\t', ' ', text)
    
    # Özel karakterlerin silinmesi
    #text = re.sub(r'.*[@_!#$%^&*()<>?/\|}{~:].*', '', text)
    text = re.sub(r'[!@#$%^&*()_+-={}[]|:;"\'<>,.?/~\]', ' ', text)
    
    # Metin verileri sayısal verilere dönüştürülecek (?)
    
    return text

# Etkisiz kelimelerin silinmesi
def remove_stop_words(text):
    return " ".join([word for word in str(text).split() if word not in stop_words])

# Stemming / Lemmatization (kelime köküne inilmesi)
lemmatizer=WordNetLemmatizer()
def stemming_lemmatization(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

news['title'] = news['title'].apply(word_operations)
news['text'] = news['text'].apply(word_operations)
news['title'] = news['title'].apply(lambda x : remove_stop_words(x))
news['text'] = news['text'].apply(lambda x : remove_stop_words(x))
news['title'] = news['title'].apply(lambda x : stemming_lemmatization(x))
news['text'] = news['text'].apply(lambda x : stemming_lemmatization(x))
print(news.head(10))










