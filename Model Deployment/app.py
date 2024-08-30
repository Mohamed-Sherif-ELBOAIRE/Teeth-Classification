import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = load_model("D:/Mohamed Sheriff/Projects/Computer Vision Internship - Cellula Technologies/Teeth Classification/Model/VGG16.h5")

# Define the class labels
class_labels = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']

st.title('Teeth Classification App')

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Load and preprocess the image
    img = load_img(uploaded_file, target_size=(128,128))  
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  

    # Predict the class
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])

    # Display the image
    st.image(img, caption=f"Uploaded Image:", use_column_width=True)
    st.write(f"Predicted Class: {class_labels[predicted_class]}")

    actual_class = st.selectbox("Actual Class", class_labels)

    if actual_class:
        st.write(f"Actual Class: {actual_class}")

        # Compare predicted and actual class
        if actual_class == class_labels[predicted_class]:
            st.success("The model's prediction matches the actual class!")
        else:
            st.error("The model's prediction does not match the actual class!")
