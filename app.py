import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('happy_model.h5')

# Streamlit app
st.title('Happy or Not Happy Classifier')
st.write('Upload an image and the model will classify it as happy or not happy.')

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    img = img.resize((64, 64))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the image

    # Make prediction
    prediction = model.predict(img)
    if prediction < 0.5:
        st.write("The model predicts: Not Happy 😟")
    else:
        st.write("The model predicts: Happy 😊")
