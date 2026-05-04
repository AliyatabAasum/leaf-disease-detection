import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import sys

print("=" * 50)
print("   LEAF DISEASE DETECTION - MODEL TESTING")
print("=" * 50)

# Load model
if not os.path.exists('model.h5'):
    print("ERROR: model.h5 not found! Run train_model.py first.")
    sys.exit()

model = load_model('model.h5')
print("✅ Model loaded successfully!")

# Class labels - must match train_data.class_indices order
class_labels = [
    'Bacterial_Blight',
    'Downy_Mildew',
    'Healthy',
    'Powdery_Mildew',
    'Rust'
]

def test_single_image(img_path):
    if not os.path.exists(img_path):
        print(f"ERROR: Image not found at {img_path}")
        return

    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100

    print(f"\nImage           : {img_path}")
    print(f"Predicted Class : {class_labels[predicted_class]}")
    print(f"Confidence      : {confidence:.2f}%")
    print(f"All predictions : {dict(zip(class_labels, prediction[0]*100))}")

def test_all_from_folder(folder):
    print(f"\nTesting all images in: {folder}")
    count = 0
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            test_single_image(os.path.join(folder, fname))
            count += 1
    print(f"\nTotal images tested: {count}")

# ─── Run Test ─────────────────────────────────────────────
print("\nHow to use:")
print("1. Test single image: test_single_image('path/to/image.jpg')")
print("2. Test folder: test_all_from_folder('dataset/test/Healthy')")
print()

# Example - change this path to your image
test_folder = 'dataset/test'
if os.path.exists(test_folder):
    for class_folder in os.listdir(test_folder):
        full_path = os.path.join(test_folder, class_folder)
        if os.path.isdir(full_path):
            images = [f for f in os.listdir(full_path)
                      if f.lower().endswith(('.jpg','.jpeg','.png'))]
            if images:
                test_single_image(os.path.join(full_path, images[0]))
else:
    print("No test images found. Add images to dataset/test/ folders.")
