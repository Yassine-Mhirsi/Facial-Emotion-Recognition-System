import streamlit as st # type: ignore
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
import cv2 # type: ignore
import os
from PIL import Image # type: ignore
import random
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense # type: ignore
import random

@st.cache_resource
def load_emotion_model():
    # Define the architecture based on your training code
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))

    # Output layer
    model.add(Dense(7, activation='softmax'))

    # Load weights from file
    model.load_weights("model.weights.h5")
    return model

# Load the model
model = load_emotion_model()


def model_prediction(image):
    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    else:
        img = np.array(image)
    img = cv2.resize(img, (48, 48))
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    input_arr = np.expand_dims(img, axis=0)
    predictions = model.predict(input_arr)
    predicted_index = np.argmax(predictions)
    predicted_prob = predictions[0][predicted_index] * 100
    class_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    return class_labels[predicted_index], predicted_prob

st.sidebar.title("Emotion Recognition Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction"])

if app_mode == "Home":
    st.header("Facial Expression Recognition")
    st.subheader("Classifying facial expressions into seven emotion categories:")
    st.image("dataset-cover.png", caption="Example of emotions classified: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral")

elif app_mode == "About Project":
    st.header("About Project")
    st.subheader("Dataset Overview")
    st.text("""The dataset consists of 48x48 pixel grayscale images of faces with labels 
for each of the seven emotions:""")
    st.code("0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral")
    st.subheader("Content")
    st.text("Training Set: 28,709 examples\nPublic Test Set: 3,589 examples")
    st.text("The task is to classify each face into one of the seven emotions.")
    st.subheader("Task")
    st.text("""The primary task is to recognize the emotion displayed in each facial image with
high accuracy.""")
    st.subheader("Model Information")
    st.text("The model is a Convolutional Neural Network (CNN) with multiple convolutional layers")
    st.text("""batch normalization, max-pooling, and dropout layers to enhance feature extraction 
and reduce overfitting.""")
    st.text("Final layer uses softmax activation to predict one of the seven emotions.")
    st.code("""
1. Data Preprocessing:
   - Grayscale images are resized to 48x48 pixels.
   - Images are normalized to enhance model performance.

2. Model Architecture:
   - Uses convolutional and pooling layers with dropout for regularization.
   - Fully connected layer with 7 outputs and softmax activation for 
     emotion classification.

3. Training & Evaluation:
   - Model is trained with 28,709 examples and validated with 3,589 examples.
   - Evaluated using accuracy and loss metrics to improve performance.

4. Prediction Interface:
   - Built using Streamlit.
   - Allows selecting an image to predict the displayed emotion.
   - Displays model prediction with confidence level.
""")

elif app_mode == "Prediction":
    st.header("Emotion Prediction")
    st.subheader("Select an emotion class to display images for prediction:")
    emotion_classes = ["Custom Image", "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    selected_class = st.selectbox("Choose an emotion class or upload a custom image", emotion_classes)
    test_folder = f"data/test/{selected_class}" if selected_class != "Custom Image" else None

    if selected_class == "Custom Image":
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        
        if uploaded_image is not None:
            img = Image.open(uploaded_image).convert("L").resize((300, 300))
            st.image(img, caption="Uploaded Image", use_column_width=False)

            if st.button("Predict"):
                predicted_label, predicted_prob = model_prediction(img)
                st.success(f"Model predicts: {predicted_label}")

    else:
        if "current_class" not in st.session_state or st.session_state.current_class != selected_class:
            st.session_state.current_class = selected_class
            image_files = [f for f in os.listdir(test_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.jfif'))]
            st.session_state.selected_images = random.sample(image_files, min(10, len(image_files)))

        if not st.session_state.selected_images:
            st.error(f"No images found in the '{selected_class}' folder.")
        else:
            num_columns = 5
            cols = st.columns(num_columns)
            display_size = (150, 150)

            for index, image_file in enumerate(st.session_state.selected_images):
                col = cols[index % num_columns]
                image_path = os.path.join(test_folder, image_file)
                with col:
                    img = Image.open(image_path)
                    img = img.resize(display_size)
                    st.image(img, use_column_width=True)

                    if st.button("Predict", key=f"predict_{image_file}"):
                        predicted_label, predicted_prob = model_prediction(image_path)
                        st.success(f"Model predicts: {predicted_label}")
