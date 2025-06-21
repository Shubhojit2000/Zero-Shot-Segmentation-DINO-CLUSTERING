# Zero-Shot Semantic Segmentation with DINO and Clustering

This repository presents a lightweight, **zero-shot segmentation pipeline** using a pretrained DINO Vision Transformer and unsupervised clustering. The approach requires no labeled data or additional training, making it ideal for rapid prototyping on new domains or classes.

---

## 📌 Overview

This method segments objects in images **without any supervision** using:

- A pretrained DINO ViT-S/8 model to extract dense patch embeddings
- Principal Component Analysis (PCA) to reduce feature dimensionality
- K-Means clustering to group semantically similar pixels
- Upsampling to generate full-resolution segmentation masks

---

## 🧠 Key Features

- 🚫 **No training required** – everything is zero-shot
- 🎯 Based on semantically rich **DINO embeddings**
- ⚡ Fast execution on CPU or GPU
- 🖼️ Saves both raw segmentation masks and visualization comparisons

---

## 🖥️ File Structure

- main.py # Main script
-  *.png # Input images (in current directory)
-   visualizations/ # Side-by-side original+mask
-   README.md # This file

---

## 🚀 How to Run

**Install dependencies**:

```bash
pip install torch torchvision matplotlib scikit-learn pillow
```

**Run the script**

```bash
python main.py
