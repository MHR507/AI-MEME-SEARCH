import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

@st.cache_data
def load_embeddings():
    embeddings = np.load("embeddings/image_embeddings.npy")
    image_paths = np.load("embeddings/image_paths.npy")
    return embeddings, image_paths

model, processor = load_model()
embeddings, image_paths = load_embeddings()

def search_images(query, top_k=5):
    text_inputs = processor(text=[query], return_tensors="pt").to(device)
    text_emb = model.get_text_features(**text_inputs)
    text_emb = text_emb / text_emb.norm(p=2)
    text_emb = text_emb.cpu().detach().numpy()

    sims = cosine_similarity(text_emb, embeddings)[0]
    idx = np.argsort(sims)[::-1][:top_k]

    results = [(image_paths[i], sims[i]) for i in idx]
    return results

st.set_page_config(page_title="Meme Search AI", layout="wide")
st.title("CLIP Meme Search Engine")
st.write("Type a text description to find the most relevant meme.")

query = st.text_input("Enter your search text:", "")

if st.button("Search") and query.strip() != "":
    with st.spinner("Searching..."):
        results = search_images(query, top_k=6)

    st.subheader(f"Top Results for: **{query}**")

    cols = st.columns(3)

    for i, (path, score) in enumerate(results):
        img = Image.open(path).convert("RGB")
        with cols[i % 3]:
            st.image(img, caption=f"Score: {score:.4f}", use_container_width=True)
