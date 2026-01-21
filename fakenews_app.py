import warnings
warnings.filterwarnings("ignore")

import torch
torch.set_num_threads(1)

import re
import requests
import streamlit as st
from bs4 import BeautifulSoup
from PIL import Image

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    ViTImageProcessor,
    ViTForImageClassification
)

st.set_page_config(page_title="Fake News Detection System", layout="wide")

# ================= TEXT MODEL =================

@st.cache_resource
def load_text_model():
    model_name = "mrm8488/bert-tiny-finetuned-fake-news-detection"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

# ================= IMAGE MODEL =================
# (ImageNet-trained – demo / future scope)

@st.cache_resource
def load_image_model():
    model_name = "google/vit-base-patch16-224"
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)
    model.eval()
    return processor, model

# ================= UTILS =================

def clean_text(text):
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    return text.lower().strip()

def scrape_article(url):
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "lxml")
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
        return text if len(text.split()) > 50 else None
    except:
        return None

def predict_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    return pred

def predict_image(image, processor, model):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    return pred

# ================= UI =================

def main():
    st.title("📰 Fake News Detection System")

    tab1, tab2, tab3 = st.tabs(["📝 Text", "🔗 URL", "🖼 Image"])

    # -------- TEXT --------
    with tab1:
        text_input = st.text_area("Paste news text", height=200)
        if st.button("Predict Text"):
            if not text_input.strip():
                st.warning("Enter text")
            else:
                tokenizer, model = load_text_model()
                pred = predict_text(clean_text(text_input), tokenizer, model)
                st.success("REAL" if pred == 1 else "FAKE")

    # -------- URL --------
    with tab2:
        url = st.text_input("Enter article URL")
        if st.button("Predict URL"):
            article = scrape_article(url)
            if article:
                tokenizer, model = load_text_model()
                pred = predict_text(clean_text(article), tokenizer, model)
                st.success("REAL" if pred == 1 else "FAKE")
            else:
                st.error("Could not extract article")

    # -------- IMAGE --------
    with tab3:
        st.info(
            "Image model demonstrates Vision Transformer usage.\n"
            "True fake-image detection requires fine-tuning on misinformation datasets."
        )

        uploaded = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, use_column_width=True)

            if st.button("Analyze Image"):
                processor, model = load_image_model()
                pred = predict_image(image, processor, model)
                st.write("Image analyzed using ViT (feature-level inference)")

# ================= RUN =================

if __name__ == "__main__":
    main()
