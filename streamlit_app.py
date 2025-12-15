import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    clean = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean

model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# Making the page look better
st.set_page_config(page_title="Spam Classifier", layout="centered")

st.title("Spam Email Classifier")
st.write("Check if your email is safe or spam.")

email_text = st.text_area("Paste Email Content:", height=200)

if st.button("Check Spam"):
    if not email_text:
        st.warning("Please type something.")
    else:
        # showing the features to the user
        length = len(email_text)
        exclam = email_text.count('!')
        
        col1, col2 = st.columns(2)
        col1.metric("Msg Length", str(length))
        col2.metric("Exclamation Marks", str(exclam))
        
        st.markdown("---")

        vec_text = vectorizer.transform([email_text])
        prediction = model.predict(vec_text)[0]
        
        if prediction == 1:
            st.error("SPAM DETECTED")
        else:
            st.success("NOT SPAM")