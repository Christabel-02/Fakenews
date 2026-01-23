# ---- FULL APP CODE START ----

import warnings
warnings.filterwarnings("ignore")

import torch
torch.set_num_threads(1)

import requests
from bs4 import BeautifulSoup
import re
import streamlit as st
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    ViTImageProcessor,
    ViTForImageClassification
)
from PIL import Image

st.set_page_config(page_title="Fake News Detector", layout="wide")


# ----------------- TEXT MODEL -----------------
@st.cache_resource
def load_text_model():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


# ----------------- IMAGE MODEL -----------------
@st.cache_resource
def load_image_model():
    model_name = "facebook/deit-small-patch16-224"
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)
    model.eval()
    return processor, model


# ----------------- UTILS -----------------
def scrape_article(url):
    try:
        resp = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"}
        )
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
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=-1).item()
    return pred


def predict_image(image, processor, model):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=-1).item()
    return pred


# ----------------- STREAMLIT APP -----------------
def main():
    st.title("📰 Fake News Detection System")

    tab1, tab2, tab3 = st.tabs(
        ["📝 Text", "🔗 URL", "🖼 Image"]
    )

    with tab1:
        text_input = st.text_area("Paste article text:")
        if st.button("Predict Text"):
            tokenizer, model = load_text_model()
            pred = predict_text(clean_text(text_input), tokenizer, model)
            st.success("REAL" if pred == 1 else "FAKE")

    with tab2:
        url = st.text_input("Enter URL:")
        if st.button("Predict URL"):
            article = scrape_article(url)
            if article:
                tokenizer, model = load_text_model()
                pred = predict_text(clean_text(article), tokenizer, model)
                st.success("REAL" if pred == 1 else "FAKE")
            else:
                st.error("Could not extract article")

    with tab3:
        uploaded = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image)
            if st.button("Predict Image"):
                processor, model = load_image_model()
                pred = predict_image(image, processor, model)
                st.success("REAL" if pred % 2 == 0 else "FAKE")


if __name__ == "__main__":
    main()

# ---- FULL APP CODE END ----
