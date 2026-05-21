# Kidney Tumor Detection (3D-CNN & Transformer)

![Healthcare Technology](https://img.shields.io/badge/Domain-Healthcare-red?style=for-the-badge)
![AI/ML](https://img.shields.io/badge/Stack-AI/ML-blue?style=for-the-badge)
![Flask](https://img.shields.io/badge/Framework-Flask-lightgrey?style=for-the-badge)

A state-of-the-art medical imaging application that leverages 3D Convolutional Neural Networks (3D-CNN) and Transformers to detect and classify kidney tumors from CT scans.

## 🚀 Overview

This project provides a robust solution for automated kidney tumor analysis. It consists of a high-performance backend for model inference and a modern, user-friendly frontend for radiologists and medical professionals to upload scans and view results.

## 🧠 Model Weights

> [!IMPORTANT]
> Due to the large size (~1.2 GB), the pre-trained model weights are not hosted directly on GitHub.
> 
> **[Download Model Weights (Placeholder Link)]**
> *Instructions: Download the `.pth` files and place them in the `BACKEND/` and `FRONTEND/` directories.*

## 🛠️ Technology Stack

- **Deep Learning**: PyTorch, 3D-CNN, Vision Transformers
- **Backend**: Python, Flask
- **Frontend**: HTML5, CSS3 (Modern Glassmorphic UI), JavaScript
- **Image Processing**: OpenCV, Nibabel (for NIfTI files)

## 📦 Project Structure

```text
├── BACKEND/          # AI Model logic and training notebooks
├── FRONTEND/         # Flask application and UI templates
├── static/           # Global assets (CSS, images)
└── .gitignore        # Excludes large binaries and datasets
```

## 🔧 Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/raamprasanth21/kidney-tumor-detection-3dcnn-transformer.git
   cd kidney-tumor-detection-3dcnn-transformer
   ```

2. **Install dependencies:**
   ```bash
   pip install -r FRONTEND/requirements.txt
   ```

3. **Run the application:**
   ```bash
   cd FRONTEND
   python app.py
   ```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
