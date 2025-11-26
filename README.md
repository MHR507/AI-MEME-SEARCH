# AI-MEME-SEARCH  
A powerful AI-based meme and image search engine using **CLIP (ViT)** and **Stable Diffusion** text-to-image generation.

This project allows you to:
- 🔍 Search memes using **text descriptions**  
- 🧠 Use **CLIP** (Contrastive Language–Image Pretraining) to find the most similar images  
- 🎨 Optionally generate **new AI images** using Stable Diffusion  
- ⚡ Supports GPU acceleration for fast inference  
- 🖥️ Easy-to-use **Streamlit UI**

---

# 🚀 Features

### ✔ Meme/Image Search (CLIP)
Enter a text query like:
> "funny confused cat"

The system:
1. Converts text → embedding
2. Compares with all image embeddings
3. Returns top matching memes

### ✔ AI Image Generation (Stable Diffusion)
Enter a prompt like:
> "cat hacking a computer meme"

The app generates a **new AI image** instantly.

### ✔ Fast GPU support
Uses:
- PyTorch CUDA  
- Xformers optimization  
- Streamlit UI  

---

# 🧠 Tech Stack

| Component | Technology |
|----------|------------|
| Model | CLIP (ViT-B/32) |
| Generator (optional) | Stable Diffusion v1.5 |
| UI | Streamlit |
| Backend | Python |
| Embeddings | NumPy |
| Similarity | Cosine Similarity |
| Image processing | Pillow |

---

# 📦 Installation & Setup

## 1️⃣ Clone repository
```bash
git clone https://github.com/yourusername/AI-MEME-SEARCH.git
cd AI-MEME-SEARCH
```
## 2️⃣ Create Conda Environment (Recommended)
```bash
conda create -n meme_ai python=3.10 -y
conda activate meme_ai
```
## 3️⃣ Install PyTorch (GPU or CPU)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
#if you dont have a gpu then
pip install torch torchvision
```
## 4️⃣ Install Required Python Packages
```bash
pip install requirements.txt
```