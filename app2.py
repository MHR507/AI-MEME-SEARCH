import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# ---------------------------
# Setup Device
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# Load Stable Diffusion Model
# ---------------------------
@st.cache_resource
def load_sd():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    if device == "cuda":
        pipe.to("cuda")
    else:
        pipe.to("cpu")

    return pipe

pipe = load_sd()

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Stable Diffusion Generator", layout="wide")
st.title("Stable Diffusion Image Generator")
st.write("Generate images using Stable Diffusion.")

prompt = st.text_input("Enter your prompt:", "funny cat meme confused")
num_images = st.slider("Number of images", 1, 4, 2)

if st.button("Generate"):
    with st.spinner("Generating..."):
        images = pipe(
            prompt,
            num_inference_steps=30,
            num_images_per_prompt=num_images
        ).images

    st.subheader("Generated Output")
    cols = st.columns(2)

    for i, img in enumerate(images):
        with cols[i % 2]:
            st.image(img, use_container_width=True)
