
import warnings
warnings.filterwarnings("ignore")

import torch
torch.set_num_threads(1)

import streamlit as st
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image

st.set_page_config(
    page_title="Fake Image Detection",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_vit_model():
    model_path = "./vit_fake_real"  # folder containing trained model
    processor = ViTImageProcessor.from_pretrained(model_path)
    model = ViTForImageClassification.from_pretrained(
        model_path,
        num_labels=2
    )
    model.eval()
    return processor, model

# ---------------- PREDICTION ----------------

def predict_image(image, processor, model):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
    return pred, confidence

# ---------------- UI ----------------

def main():
    st.title("🖼 Fake Image Detection System")
    st.write(
        "This system uses a **Vision Transformer (ViT)** "
        "trained to classify images as **FAKE** or **REAL**."
    )

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            with st.spinner("Analyzing image..."):
                processor, model = load_vit_model()
                pred, conf = predict_image(image, processor, model)

            if pred == 0:
                st.error(f"🟥 FAKE IMAGE\n\nConfidence: {conf:.2%}")
            else:
                st.success(f"🟩 REAL IMAGE\n\nConfidence: {conf:.2%}")

# ---------------- RUN ----------------

if __name__ == "__main__":
    main()
