========================================================
   LEAF DISEASE DETECTION SYSTEM
   Malla Reddy Engineering College | CSE-AIML | 2024-25
   Team: Aliya Tabasum, Niveditha, Madhu, Mahendra
========================================================

STEP 1 - SETUP VIRTUAL ENVIRONMENT
------------------------------------
Open Command Prompt (cmd) and run:

    py -3.11 -m venv leaf_env
    leaf_env\Scripts\activate

You will see (leaf_env) appear in terminal.


STEP 2 - INSTALL LIBRARIES
------------------------------------
    pip install -r requirements.txt


STEP 3 - ADD DATASET IMAGES
------------------------------------
Download PlantVillage dataset from:
https://www.kaggle.com/datasets/emmarex/plantdisease

Add images to these folders:
    dataset/train/Healthy/
    dataset/train/Powdery_Mildew/
    dataset/train/Downy_Mildew/
    dataset/train/Bacterial_Blight/
    dataset/train/Rust/
    dataset/test/Healthy/
    dataset/test/Powdery_Mildew/
    (same for other classes)

Minimum: 50+ images per class for training.


STEP 4 - TRAIN THE MODEL
------------------------------------
    python train_model.py

This creates model.h5 file. Takes 5-15 minutes.


STEP 5 - TEST THE MODEL (Optional)
------------------------------------
    python test_model.py


STEP 6 - RUN THE GUI APPLICATION
------------------------------------
    python app.py


HOW TO USE THE APP
------------------------------------
1. Click "Upload Image" - select a leaf photo
2. Click "Predict" - see the disease result
3. Click "Reset" - clear and start again


DISEASE CLASSES
------------------------------------
- Healthy
- Powdery Mildew
- Downy Mildew
- Bacterial Blight
- Rust


FILES IN THIS PROJECT
------------------------------------
leaf-disease-detection/
├── dataset/
│   ├── train/        <- Add training images here
│   └── test/         <- Add test images here
├── train_model.py    <- Run this FIRST to train
├── test_model.py     <- Run this to test accuracy
├── app.py            <- Main GUI application
├── requirements.txt  <- All library versions
├── model.h5          <- Generated after training
└── README.txt        <- This file


TROUBLESHOOTING
------------------------------------
Q: tensorflow not installing?
A: Make sure virtual env is activated (leaf_env\Scripts\activate)
   Then: pip install tensorflow==2.13.0

Q: model.h5 not found when running app.py?
A: Run train_model.py first!

Q: Wrong predictions?
A: Add more images (100+ per class) and retrain.

Q: Class labels mismatch?
A: Check class_indices printed during training.
   Update class_labels list in app.py to match.
