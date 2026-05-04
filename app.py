import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# ─── 1. Load Model ───────────────────────────────────────
MODEL_PATH = 'model.h5'

if not os.path.exists(MODEL_PATH):
    import tkinter as tk
    from tkinter import messagebox
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(
        "Model Not Found",
        "model.h5 not found!\nPlease run train_model.py first."
    )
    exit()

print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# ─── 2. Class Labels ─────────────────────────────────────
# Order must match train_data.class_indices from training
class_labels = [
    'Bacterial_Blight',
    'Downy_Mildew',
    'Healthy',
    'Powdery_Mildew',
    'Rust'
]

disease_info = {
    'Healthy':
        '✅ The leaf is HEALTHY!\nNo treatment needed. Keep monitoring regularly.',
    'Powdery_Mildew':
        '⚠️ POWDERY MILDEW detected!\nCaused by fungus.\nTreatment: Apply sulfur-based fungicide.',
    'Downy_Mildew':
        '⚠️ DOWNY MILDEW detected!\nCaused by water mold.\nTreatment: Improve air circulation, reduce moisture.',
    'Bacterial_Blight':
        '🚨 BACTERIAL BLIGHT detected!\nBacterial infection.\nTreatment: Remove infected leaves, apply copper spray.',
    'Rust':
        '🚨 RUST disease detected!\nFungal disease.\nTreatment: Apply copper-based fungicide immediately.'
}

# ─── 3. Predict Function ─────────────────────────────────
def predict_disease(img_path):
    try:
        img = image.load_img(img_path, target_size=(64, 64))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction) * 100
        label = class_labels[predicted_class]
        info = disease_info.get(label, 'No information available.')
        return label, confidence, info
    except Exception as e:
        return f"Error: {str(e)}", 0, ''

# ─── 4. GUI Functions ─────────────────────────────────────
file_path_global = None

def upload_image():
    global file_path_global
    file_path = filedialog.askopenfilename(
        title="Select Leaf Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        return
    file_path_global = file_path

    img = Image.open(file_path).resize((220, 220))
    img_tk = ImageTk.PhotoImage(img)
    img_label.config(image=img_tk, text='')
    img_label.image = img_tk

    result_label.config(text='Disease: —', fg='white')
    confidence_label.config(text='Confidence: —')
    info_label.config(text='Upload successful! Click Predict.')
    predict_btn.config(state='normal')

def predict():
    if not file_path_global:
        messagebox.showwarning("No Image", "Please upload a leaf image first!")
        return

    info_label.config(text='⏳ Analyzing image...')
    root.update()

    disease, confidence, info = predict_disease(file_path_global)

    color = '#00b894' if disease == 'Healthy' else '#d63031'
    display_name = disease.replace('_', ' ')

    result_label.config(text=f"Disease: {display_name}", fg=color)
    confidence_label.config(text=f"Confidence: {confidence:.2f}%")
    info_label.config(text=info)

def reset_all():
    global file_path_global
    file_path_global = None
    img_label.config(image='', text='📷 No Image Selected', bg='#353b48')
    img_label.image = None
    result_label.config(text='Disease: —', fg='white')
    confidence_label.config(text='Confidence: —')
    info_label.config(text='')
    predict_btn.config(state='disabled')

# ─── 5. Build GUI ─────────────────────────────────────────
root = tk.Tk()
root.title("🌿 Leaf Disease Detection System")
root.geometry("520x680")
root.configure(bg='#2d3436')
root.resizable(False, False)

# ── Header ──
header_frame = tk.Frame(root, bg='#00b894', pady=12)
header_frame.pack(fill='x')

tk.Label(
    header_frame,
    text="🌿 Leaf Disease Detection System",
    font=("Arial", 17, "bold"),
    bg='#00b894', fg='white'
).pack()

tk.Label(
    header_frame,
    text="Malla Reddy Engineering College | CSE-AIML | 2024-25",
    font=("Arial", 8),
    bg='#00b894', fg='#dfe6e9'
).pack()

# ── Image Box ──
img_frame = tk.Frame(root, bg='#353b48', bd=2, relief='groove')
img_frame.pack(pady=15, padx=20)

img_label = tk.Label(
    img_frame,
    text="📷 No Image Selected",
    width=30, height=12,
    bg='#353b48', fg='#b2bec3',
    font=("Arial", 12)
)
img_label.pack(padx=10, pady=10)

# ── Buttons ──
btn_frame = tk.Frame(root, bg='#2d3436')
btn_frame.pack(pady=5)

tk.Button(
    btn_frame,
    text="📁  Upload Image",
    command=upload_image,
    font=("Arial", 12, "bold"),
    bg='#0984e3', fg='white',
    width=14, pady=6,
    relief='flat', cursor='hand2',
    activebackground='#74b9ff'
).grid(row=0, column=0, padx=8)

predict_btn = tk.Button(
    btn_frame,
    text="🔍  Predict",
    command=predict,
    font=("Arial", 12, "bold"),
    bg='#00b894', fg='white',
    width=10, pady=6,
    relief='flat', cursor='hand2',
    activebackground='#55efc4',
    state='disabled'
)
predict_btn.grid(row=0, column=1, padx=8)

tk.Button(
    btn_frame,
    text="🔄  Reset",
    command=reset_all,
    font=("Arial", 12, "bold"),
    bg='#d63031', fg='white',
    width=10, pady=6,
    relief='flat', cursor='hand2',
    activebackground='#ff7675'
).grid(row=0, column=2, padx=8)

# ── Divider ──
tk.Frame(root, bg='#636e72', height=1).pack(fill='x', padx=20, pady=12)

# ── Results ──
result_frame = tk.Frame(root, bg='#2d3436')
result_frame.pack(pady=5)

result_label = tk.Label(
    result_frame,
    text="Disease: —",
    font=("Arial", 16, "bold"),
    bg='#2d3436', fg='white'
)
result_label.pack(pady=4)

confidence_label = tk.Label(
    result_frame,
    text="Confidence: —",
    font=("Arial", 12),
    bg='#2d3436', fg='#b2bec3'
)
confidence_label.pack(pady=3)

# ── Info Box ──
info_frame = tk.Frame(root, bg='#353b48', bd=1, relief='groove')
info_frame.pack(padx=20, pady=10, fill='x')

info_label = tk.Label(
    info_frame,
    text="",
    font=("Arial", 10),
    bg='#353b48', fg='#fdcb6e',
    wraplength=450,
    justify='center',
    pady=10
)
info_label.pack(padx=10)

# ── Footer ──
tk.Label(
    root,
    text="Automated Detection of Leaf Diseases Using Deep Learning",
    font=("Arial", 8),
    bg='#2d3436', fg='#636e72'
).pack(side='bottom', pady=8)

root.mainloop()
