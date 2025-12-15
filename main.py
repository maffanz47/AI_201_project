import pandas as pd
import string
import nltk
import os
from nltk.corpus import stopwords
from pickle import dump
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords')

dataset = pd.read_csv('emails.csv')
dataset.drop_duplicates(inplace=True)

def process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    clean = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean

# 3. Vectorization
# converting text to numbers
vectorizer = CountVectorizer(analyzer=process)
X = vectorizer.fit_transform(dataset['text'])
y = dataset['spam']

# create models folder if not there
os.makedirs("models", exist_ok=True)

# saving vectorizer
dump(vectorizer, open("models/vectorizer.pkl", "wb"))

# 4. Splitting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# 5. Training
model = MultinomialNB()
model.fit(X_train, y_train)

# save the model
dump(model, open("models/model.pkl", "wb"))

# 6. Evaluation
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")