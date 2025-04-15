import streamlit as st
import pandas as pd
import requests
import re
import textstat
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

st.set_page_config(page_title="WordTrain – Difficulty Classifier", layout="centered")
st.title("📚 WordTrain – Real-Time Word Difficulty Classifier")

# ========== STEP 1: Input ==========
st.markdown("### 🔗 Provide a text link (e.g., from Google Lens output)")

url = st.text_input("Paste URL of plain-text file", placeholder="https://...")

if url and st.button("🔄 Fetch and Train Model"):
    try:
        response = requests.get(url)
        raw_text = response.text
        words = re.findall(r'\b[a-zA-Z]{2,}\b', raw_text.lower())
        unique_words = list(set(words))
        st.success(f"✅ Fetched {len(unique_words)} unique words!")

        # ========== STEP 2: Feature Engineering ==========
        def extract_features(word_list):
            data = []
            for w in word_list:
                data.append({
                    "word": w,
                    "length": len(w),
                    "syllables": textstat.syllable_count(w),
                    "frequency": textstat.lexicon_count(w)
                })
            return pd.DataFrame(data)

        df = extract_features(unique_words)

        # ========== STEP 3: Auto-label Difficulty ==========
        def label(row):
            if row["length"] <= 4 and row["syllables"] <= 1:
                return "Easy"
            elif row["length"] <= 7:
                return "Medium"
            return "Hard"

        df["difficulty"] = df.apply(label, axis=1)

        # ========== STEP 4: Train ==========
        X = df[["length", "syllables", "frequency"]]
        y = LabelEncoder().fit_transform(df["difficulty"])
        model = RandomForestClassifier().fit(X, y)
        joblib.dump(model, "word_difficulty_model.pkl")
        st.success("🎉 Model trained and saved successfully!")

    except Exception as e:
        st.error(f"❌ Failed to process link: {e}")

# ========== STEP 5: Word Prediction ==========
st.markdown("---")
st.markdown("### 🔤 Type a word to check its difficulty")

word_input = st.text_input("Enter a word")

if word_input:
    if os.path.exists("word_difficulty_model.pkl"):
        model = joblib.load("word_difficulty_model.pkl")
        features = [[
            len(word_input),
            textstat.syllable_count(word_input),
            textstat.lexicon_count(word_input)
        ]]
        prediction = model.predict(features)[0]
        label_map = {0: "Easy", 1: "Hard", 2: "Medium"}  # Update if needed
        predicted_label = label_map.get(prediction, "Unknown")

        st.info(f"📘 **{word_input}** is classified as: **{predicted_label}**")
    else:
        st.warning("⚠️ No trained model found. Please train one first.")
