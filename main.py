import os
import pandas as pd
import numpy as np  # Used for stats
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
from nltk.corpus import stopwords
from pickle import dump
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Download NLTK stopwords
nltk.download('stopwords')

dataset = pd.read_csv(
    'C:\\Users\\Affan\\Desktop\\Email-Spam-Detection-Using-Machine-Learning\\emails.csv')

dataset.drop_duplicates(inplace=True)

# --- NEW FEATURE 1: Message Length ---
dataset['message_len'] = dataset['text'].apply(len)

# --- NEW FEATURE 2: '!' Punctuation Count ---
dataset['exclam_count'] = dataset['text'].apply(lambda x: x.count('!'))

# --- VISUALIZATION (Using Matplotlib & Numpy) ---
# 1. Compare Average '!' count using Numpy
avg_spam_exclam = np.mean(dataset[dataset['spam'] == 1]['exclam_count'])
avg_ham_exclam = np.mean(dataset[dataset['spam'] == 0]['exclam_count'])
print(f"Average '!' in Spam: {avg_spam_exclam:.2f}")
print(f"Average '!' in Not Spam: {avg_ham_exclam:.2f}")

# 2. Visualize Message Length Distribution
plt.figure(figsize=(10, 6))
# Plotting two histograms on the same chart
plt.hist(dataset[dataset['spam'] == 0]['message_len'],
         bins=50, alpha=0.5, label='Not Spam')
plt.hist(dataset[dataset['spam'] == 1]['message_len'],
         bins=50, alpha=0.5, color='red', label='Spam')
plt.title("Message Length Distribution (Spam vs Non-Spam)")
plt.xlabel("Message Length")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# --- MODEL TRAINING ---


def process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    clean = [word for word in nopunc.split() if word.lower()
             not in stopwords.words('english')]
    return clean


# Vectorize text data
vectorizer = CountVectorizer(analyzer=process)
X = vectorizer.fit_transform(dataset['text'])
y = dataset['spam']

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# Save the vectorizer
dump(vectorizer, open("models/vectorizer.pkl", "wb"))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save the trained model
dump(model, open("models/model.pkl", "wb"))

# Predictions & Evaluation
y_pred = model.predict(X_test)
print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(dpi=100)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[
            'Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam'])
plt.title("Confusion Matrix")
plt.show()
