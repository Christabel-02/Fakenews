import os
import requests
from bs4 import BeautifulSoup
import re
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
from PIL import Image

st.set_page_config(page_title="Fake News & Image Forgery Detector", layout="wide")

# 🔥 TEMP: clear cache (run once, then remove)
st.cache_resource.clear()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= TEXT MODEL =================
@st.cache_resource
def load_text_model():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

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

# ================= IMAGE MODEL =================
@st.cache_resource
def load_image_model():
    detector = pipeline(
        "image-classification",
        model="umm-maybe/AI-image-detector",
        device=0 if torch.cuda.is_available() else -1
    )
    return detector

def predict_image(image, detector):
    image = image.convert("RGB")
    result = detector(image)

    top = result[0]
    label_raw = top['label'].lower()
    confidence = top['score']

    if 'human' in label_raw or 'real' in label_raw:
        label = 'REAL'
    else:
        label = 'AI GENERATED'

    return label, confidence

# ================= MAIN APP =================
def main():
    st.title("📰🖼 Multi-Modal Fake Detection System")
    st.write("Detect Fake News (Text/URL) and AI Generated Images")

    # 🔥 Debug version check
    st.write("Version 2.0 🚀")

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
        st.subheader("AI Image Detection")

        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # 🔥 Load model once (not inside button)
            detector = load_image_model()

            analyze = st.button("Analyze Image")

            if analyze:
                with st.spinner("Analyzing image..."):
                    label, confidence = predict_image(image, detector)

                st.subheader("Prediction Result")

                if label == "AI GENERATED":
                    st.error(f"🤖 Prediction: {label}")
                else:
                    st.success(f"✅ Prediction: {label}")

                st.write(f"Confidence Score: {confidence*100:.2f}%")

# ================= RUN =================
if __name__ == "__main__":
    main()
