import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# Load your trained model
MODEL_PATH = 'discriminator_model.h5'
model = load_model(MODEL_PATH)

# Define your emotion classes
EMOTIONS = ['Happy', 'Sad', 'Angry', 'Surprised', 'Neutral']

def model_predict(image, model):
    # Resize the image to match model's expected input and convert to grayscale
    processed_img = ImageOps.grayscale(image.resize((32, 32), Image.Resampling.LANCZOS))
    
    # Convert the PIL image to a numpy array
    img_array = keras_image.img_to_array(processed_img) / 255.0  # Scale pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension

    # Get model predictions
    real_or_fake, emotions = model.predict(img_array)
    return emotions  # We are interested in the emotions part only

# Set up the title of the app
st.title("Emotion Detection App")

# Upload image interface
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the image with PIL
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Classifying...")
    
    # Make predictions
    preds = model_predict(image, model)
    pred_class = np.argmax(preds, axis=-1)  # Assuming preds is an array with one row per input image
    result = EMOTIONS[pred_class[0]]
    st.write(f"Predicted Emotion: {result}")
