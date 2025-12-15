import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
import os
from nltk.corpus import stopwords
from pickle import dump
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create models folder
os.makedirs("models", exist_ok=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# --- OPTIMIZATION: Load Stopwords ONCE as a Set ---
# This makes the processing 100x faster
STOPWORDS = set(stopwords.words('english'))

try:
    dataset = pd.read_csv('emails.csv')
except FileNotFoundError:
    print("Error: 'emails.csv' not found.")
    exit()

dataset.drop_duplicates(inplace=True)

# --- ANALYSIS ---
dataset['message_len'] = dataset['text'].apply(len)
dataset['exclam_count'] = dataset['text'].apply(lambda x: x.count('!'))
dataset['word_count'] = dataset['text'].apply(lambda x: len(x.split()))

avg_spam_words = np.mean(dataset[dataset['spam'] == 1]['word_count'])
avg_ham_words = np.mean(dataset[dataset['spam'] == 0]['word_count'])
print(f"Stats Analysis:")
print(f"- Avg Words in Spam: {avg_spam_words:.1f}")
print(f"- Avg Words in Normal: {avg_ham_words:.1f}")

# --- VISUALIZATION ---
plt.figure(figsize=(12, 5))

# Graph 1: Word Count
plt.subplot(1, 2, 1)
plt.hist(dataset[dataset['spam'] == 0]['word_count'],
         bins=50, alpha=0.5, label='Not Spam')
plt.hist(dataset[dataset['spam'] == 1]['word_count'],
         bins=50, alpha=0.5, color='red', label='Spam')
plt.title("Word Count Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.legend()

# Graph 2: Exclamation Count
plt.subplot(1, 2, 2)
sns.barplot(x='spam', y='exclam_count', data=dataset,
            hue='spam', palette='viridis', legend=False)
plt.title("Avg Exclamation Marks per Email")
plt.xticks([0, 1], ['Not Spam', 'Spam'])

plt.tight_layout()
plt.show()

# --- OPTIMIZED PROCESSING ---


def process(text):
    # 1. Remove Punctuation (Fast way)
    nopunc = text.translate(str.maketrans('', '', string.punctuation))

    # 2. Remove Stopwords (Fast way using Set)
    clean = [word for word in nopunc.split() if word.lower() not in STOPWORDS]
    return clean


print("Vectorizing data... (This should be fast now)")
vectorizer = CountVectorizer(analyzer=process)
X = vectorizer.fit_transform(dataset['text'])
y = dataset['spam']

print("Saving Vectorizer...")
dump(vectorizer, open("models/vectorizer.pkl", "wb"))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0)

print("Training Model...")
model = MultinomialNB()
model.fit(X_train, y_train)

print("Saving Model...")
dump(model, open("models/model.pkl", "wb"))

# --- PERFORMANCE STATS ---
print("\nEvaluating Model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Optional: Show confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

print(" Models saved and stats calculated.")
