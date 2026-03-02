# ---- FULL APP CODE START ----

import requests
from bs4 import BeautifulSoup
import re
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Fake News & Image Forgery Detector", layout="wide")

IMG_SIZE = 224

# ---------------- TEXT MODEL ----------------
@st.cache_resource
def load_text_model():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

# ---------------- IMAGE MODEL ----------------
@st.cache_resource
def load_image_model():
    model = tf.keras.models.load_model("forgery_model.keras")
    return model

# ---------------- TEXT FUNCTIONS ----------------
def scrape_article(url):
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "lxml")

        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text(" ", strip=True) for p in paragraphs])
        return text if len(text.split()) > 50 else None
    except:
        return None

def clean_text(text):
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    return text.lower().strip()

def predict_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=-1).item()
    return pred

# ---------------- IMAGE FUNCTIONS ----------------
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_image(image, model):
    processed = preprocess_image(image)
    probability = model.predict(processed)[0][0]
    label = "REAL" if probability > 0.65 else "FAKE"
    return label, probability

# ---------------- MAIN APP ----------------
def main():
    st.title("📰🖼 Multi-Modal Fake Detection System")
    st.write("Detect Fake News (Text/URL) and Image Forgeries")

    tab1, tab2, tab3 = st.tabs(["📝 Text Detection", "🔗 URL Detection", "🖼 Image Detection"])

    # -------- TEXT TAB --------
    with tab1:
        st.subheader("Manual Text Input")
        text_input = st.text_area("Paste article text here:", height=200)

        if st.button("Predict Text"):
            if not text_input.strip():
                st.warning("Please enter some text.")
            else:
                tokenizer, model = load_text_model()
                cleaned = clean_text(text_input)
                pred = predict_text(cleaned, tokenizer, model)
                label = "FAKE" if pred == 0 else "REAL"

                if label == "FAKE":
                    st.error(f"Prediction: {label}")
                else:
                    st.success(f"Prediction: {label}")

    # -------- URL TAB --------
    with tab2:
        st.subheader("URL-Based Detection")
        url = st.text_input("Enter article URL:")

        if st.button("Scrape & Predict"):
            if not url.strip():
                st.warning("Please enter a valid URL.")
            else:
                st.info("Scraping article...")
                article = scrape_article(url)

                if not article:
                    st.error("Failed to extract article content.")
                else:
                    st.success("Scraping Successful!")
                    preview = " ".join(article.split()[:120])
                    st.text_area("Scraping Preview:", preview, height=200)
                    st.write(f"Word Count: {len(article.split())}")

                    tokenizer, model = load_text_model()
                    cleaned = clean_text(article)
                    pred = predict_text(cleaned, tokenizer, model)
                    label = "FAKE" if pred == 0 else "REAL"

                    st.subheader("Prediction Result")
                    if label == "FAKE":
                        st.error(f"Prediction: {label}")
                    else:
                        st.success(f"Prediction: {label}")

    # -------- IMAGE TAB --------
    with tab3:
        st.subheader("Image Forgery Detection")
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Analyze Image"):
                model = load_image_model()
                label, probability = predict_image(image, model)

                st.subheader("Prediction Result")

                if label == "FAKE":
                    st.error(f"Prediction: {label}")
                else:
                    st.success(f"Prediction: {label}")

                st.write(f"Confidence Score: {probability:.4f}")

if __name__ == "__main__":
    main()

# ---- FULL APP CODE END ----
