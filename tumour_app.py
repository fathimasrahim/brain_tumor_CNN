import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

# Load your trained model
model = tf.keras.models.load_model('model.h5')

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((128, 128))  # Resize to match the model input
    img_array = np.array(img)  # Convert the image to a numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = tf.keras.utils.normalize(img_array, axis=1)  # Normalize the image
    return img_array

# Function to make predictions
def make_prediction(image, model):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    if prediction > 0.5:
        return "Tumor Detected"
    else:
        return "No Tumor Detected"

# Streamlit app layout
st.title("Brain Tumor Detection from MRI Scans")

# File uploader for image browsing
uploaded_file = st.file_uploader("Choose an MRI Image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    image=image.resize((300,200))
    st.image(image, caption='Uploaded MRI Image', use_column_width=True)
    
    # Make prediction when user clicks the button
    st.write("Processing...")
    prediction = make_prediction(image, model)
    
    # Display the prediction result
    st.write(f"Prediction: *{prediction}*")
