import os
import requests
from bs4 import BeautifulSoup
import re
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
from PIL import Image
from langdetect import detect

st.set_page_config(page_title="Fake News & Image Forgery Detector", layout="wide")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= TEXT MODELS =================
@st.cache_resource
def load_english_pipeline():
    clf = pipeline(
        "text-classification",
        model="mrm8488/bert-tiny-finetuned-fake-news-detection"
    )
    return clf

@st.cache_resource
def load_multilingual_model():
    model_name = "hamzab/roberta-fake-news-classification"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

def detect_language(text):
    try:
        lang = detect(text)
        return lang
    except:
        return "en"

def scrape_article(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        resp = requests.get(url, timeout=15, headers=headers)
        soup = BeautifulSoup(resp.text, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
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

LANGUAGE_NAMES = {
    "en": "English", "fr": "French", "de": "German", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian",
    "ar": "Arabic", "zh-cn": "Chinese", "zh-tw": "Chinese", "ja": "Japanese",
    "ko": "Korean", "hi": "Hindi", "ta": "Tamil", "tr": "Turkish",
    "pl": "Polish", "sv": "Swedish", "da": "Danish", "fi": "Finnish",
}

def get_language_name(lang_code):
    return LANGUAGE_NAMES.get(lang_code, lang_code.upper())

def run_prediction(text):
    lang = detect_language(text)
    lang_name = get_language_name(lang)
    st.info(f"🌐 Detected Language: **{lang_name}**")

    cleaned = clean_text(text)

    if lang == "en":
        clf = load_english_pipeline()
        result = clf(cleaned[:512])[0]
        raw_label = result['label'].upper()
        if "FAKE" in raw_label:
            label = "FAKE"
        elif "REAL" in raw_label:
            label = "REAL"
        else:
            # fallback: LABEL_0 = FAKE, LABEL_1 = REAL
            label = "FAKE" if result['label'] == "LABEL_0" else "REAL"
    else:
        tokenizer, model = load_multilingual_model()
        pred = predict_text(cleaned, tokenizer, model)
        label = "FAKE" if pred == 0 else "REAL"

    return label

# ================= IMAGE MODEL =================
@st.cache_resource
def load_image_model():
    detector = pipeline(
        "image-classification",
        model="Organika/sdxl-detector",
        device=0 if torch.cuda.is_available() else -1
    )
    return detector

def predict_image(image, detector):
    image = image.convert("RGB")
    result = detector(image)
    top = result[0]
    label_raw = top['label'].lower()
    if 'real' in label_raw or 'human' in label_raw:
        label = 'REAL'
    else:
        label = 'AI GENERATED'
    return label

# ================= MAIN APP =================
def main():
    st.title("📰🖼 Multi-Modal Fake Detection System")
    st.write("Detect Fake News (Text/URL) and AI Generated Images")
    st.write("Version 3.0 🚀")

    tab1, tab2, tab3 = st.tabs(["📝 Text Detection", "🔗 URL Detection", "🖼 Image Detection"])

    # -------- TEXT TAB --------
    with tab1:
        st.subheader("Manual Text Input")
        text_input = st.text_area("Paste article text here:", height=200)

        if st.button("Predict Text"):
            if not text_input.strip():
                st.warning("Please enter some text.")
            else:
                with st.spinner("Detecting language and analyzing..."):
                    label = run_prediction(text_input)

                st.subheader("Prediction Result")
                if label == "FAKE":
                    st.error(f"❌ Prediction: {label}")
                else:
                    st.success(f"✅ Prediction: {label}")

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

                    with st.spinner("Detecting language and analyzing..."):
                        label = run_prediction(article)

                    st.subheader("Prediction Result")
                    if label == "FAKE":
                        st.error(f"❌ Prediction: {label}")
                    else:
                        st.success(f"✅ Prediction: {label}")

    # -------- IMAGE TAB --------
    with tab3:
        st.subheader("AI Image Detection")
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            detector = load_image_model()
            analyze = st.button("Analyze Image")

            if analyze:
                with st.spinner("Analyzing image..."):
                    label = predict_image(image, detector)

                st.subheader("Prediction Result")
                if label == "AI GENERATED":
                    st.error(f"🤖 Prediction: {label}")
                else:
                    st.success(f"✅ Prediction: {label}")

# ================= RUN =================
if __name__ == "__main__":
    main()
