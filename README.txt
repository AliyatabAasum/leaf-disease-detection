# 🌿 Leaf Disease Detection System

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange?style=flat&logo=tensorflow)
![Accuracy](https://img.shields.io/badge/Accuracy-82.81%25-brightgreen?style=flat)

## 📌 About
A desktop application that detects leaf diseases using a CNN deep learning model. Upload a leaf photo and get instant disease prediction with confidence percentage.

**Detects:** Healthy | Powdery Mildew | Rust

---

## 🛠️ Tech Stack
`Python 3.11` `TensorFlow 2.13` `Keras` `Tkinter` `Pillow` `NumPy` `Matplotlib`

---

## 🚀 How to Run

```bash
# 1. Create virtual environment
py -3.11 -m venv leaf_env
leaf_env\Scripts\activate

# 2. Install libraries
pip install -r requirements.txt

# 3. Train the model (first time only)
python train_model.py

# 4. Run the app
python app.py
```

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Training Accuracy | 82.81% |
| Test Confidence | 99.90% |
| Epochs | 15 |
| Classes | 3 |

---

## 📁 Files
| File | Purpose |
|------|---------|
| `app.py` | GUI application |
| `train_model.py` | Train CNN model |
| `test_model.py` | Test accuracy |
| `requirements.txt` | Dependencies |

