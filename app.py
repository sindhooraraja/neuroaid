import streamlit as st
import cv2
import numpy as np
from skimage import feature
from imutils import paths
import pickle
import pandas as pd
import joblib
from keras.models import load_model

# Load the trained model
model_path = "spiralModel.pkl"
with open(model_path, "rb") as model_file:
    spiralModel = pickle.load(model_file)

def quantify_image(image):
    features = feature.hog(image, orientations=9,
                           pixels_per_cell=(10, 10), cells_per_block=(2, 2),
                           transform_sqrt=True, block_norm="L1")
    return features

def predict(image):
    # Preprocess the input image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (200, 200))
    image = cv2.threshold(image, 0, 255,
                          cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    features = quantify_image(image)

    # Make predictions using the trained model
    prediction = spiralModel["classifier"].predict([features])[0]
    return prediction