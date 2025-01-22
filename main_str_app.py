import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL = tf.keras.models.load_model("models/potatoes.h5")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

st.title("Potato Leaf Disease Prediction with Deep Learning")
uploaded_file = st.file_uploader("Upload an image of a potato leaf")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    if st.button("Predict"):
        input_data = np.array(image)
        predictions = MODEL.predict(np.expand_dims(input_data, axis=0))
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = round(float(np.max(predictions)) * 100, 2)
        st.write(f"Predicted class: {predicted_class}")
        st.write(f"Confidence: {confidence}%")
