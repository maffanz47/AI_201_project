import streamlit as st
import pickle
import string
import nltk
import matplotlib.pyplot as plt  # Importing this for the app graphs
import numpy as np
from nltk.corpus import stopwords

nltk.download('stopwords')


def process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    clean = [word for word in nopunc.split() if word.lower()
             not in stopwords.words('english')]
    return clean


model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# --- APP LAYOUT ---
st.set_page_config(page_title="Spam Classifier", layout="centered")

st.title("Spam Email Classifier")
st.write("Analyze your email to check for spam patterns.")

email_text = st.text_area("Email Content", height=150)

if st.button("🔍 Analyze Email"):
    if not email_text:
        st.warning("Please type something first.")
    else:
        # --- FEATURE CALCULATIONS ---
        # 1. Word Count
        word_count = len(email_text.split())
        # 2. Char Length
        length = len(email_text)
        # 3. Exclamation Count
        exclams = email_text.count('!')

        # Display Stats Row
        col1, col2, col3 = st.columns(3)
        col1.metric("Word Count", word_count)
        col2.metric("Characters", length)
        col3.metric("(!) Count", exclams)

        st.divider()

        # --- LIVE GRAPH VISUALIZATION ---
        st.subheader("Visual Analysis")

        # Hardcoded averages based on our training data (approximate values)
        avg_spam_words = 25  # Spam is usually short/promotional or very long
        avg_ham_words = 15   # Normal emails vary

        # Creating a bar chart to compare THIS email vs Averages
        fig, ax = plt.subplots(figsize=(6, 3))
        categories = ['Your Email', 'Avg Normal', 'Avg Spam']
        values = [word_count, avg_ham_words, avg_spam_words]
        colors = ['blue', 'green', 'red']

        ax.bar(categories, values, color=colors)
        ax.set_ylabel("Word Count")
        ax.set_title("Word Count Comparison")

        # Show graph in Streamlit
        st.pyplot(fig)

        st.divider()

        # --- PREDICTION ---
        vec_text = vectorizer.transform([email_text])
        prediction = model.predict(vec_text)[0]

        if prediction == 1:
            st.error(" RESULT: SPAM DETECTED")
            st.write("This email matches patterns found in spam messages.")
        else:
            st.success("RESULT: SAFE EMAIL")
            st.write("This email looks like a normal conversation.")
