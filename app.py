import streamlit as st
import pickle

# Page config
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Sidebar
st.sidebar.title("📌 About")
st.sidebar.write("This app uses Machine Learning and NLP to detect fake news.")
st.sidebar.write("Built using TF-IDF and Logistic Regression.")

# Title
st.title("📰 Fake News Detection System")
st.write("Enter news text below to check if it is real or fake.")

st.write("---")

# Store text in session state
if "news_text" not in st.session_state:
    st.session_state.news_text = ""

# Example button (FIXED)
if st.button("✨ Try Example"):
    st.session_state.news_text = "Aliens have landed on Earth and are living secretly among humans."

# Input (CONNECTED properly)
input_text = st.text_area("✍️ Enter News Article", value=st.session_state.news_text, height=200)

# Predict
if st.button("🔍 Analyze News"):
    if input_text.strip() != "":
        vectorized_text = vectorizer.transform([input_text])
        prediction = model.predict(vectorized_text)
        prob = model.predict_proba(vectorized_text)

        confidence = max(prob[0]) * 100

        st.write("---")

        if prediction[0] == 1:
            st.success(f"✅ REAL NEWS\nConfidence: {confidence:.2f}%")
        else:
            st.error(f"❌ FAKE NEWS\nConfidence: {confidence:.2f}%")
    else:
        st.warning("⚠️ Please enter some text")

st.write("---")
st.write("Built with ❤️ using Machine Learning")