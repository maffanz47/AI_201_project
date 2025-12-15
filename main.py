import pandas as pd
import string
import nltk
from nltk.corpus import stopwords

# downloading the stopwords dict
nltk.download('stopwords')

# 1. Loading the dataset
# using the emails.csv file
dataset = pd.read_csv('emails.csv')

# checking if data loaded
print(dataset.head())

# drop duplicates to avoid bias
dataset.drop_duplicates(inplace=True)

# 2. Pre-processing function
# this removes punctuation and stopwords (like 'the', 'is')
def process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    # remove stopwords
    clean = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean

# Testing the function
print(dataset['text'].head().apply(process))