import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('tumor_model.h5')
    return model

model = load_model()

CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

CLASS_INFO = {
    'Glioma': 'A tumor that starts in the glial cells of the brain. Requires immediate medical attention.',
    'Meningioma': 'A tumor that forms on the membranes surrounding the brain. Usually slow growing.',
    'No Tumor': 'No tumor detected in this MRI scan.',
    'Pituitary': 'A tumor in the pituitary gland at the base of the brain. Often treatable.'
}

# Page config
st.set_page_config(
    page_title="NeuraSight",
    page_icon="🧠",
    layout="centered"
)

# Header
st.title("🧠 NeuraSight")
st.subheader("AI-Powered Brain Tumor Detection & Analysis")
st.markdown("---")

# Upload
uploaded_file = st.file_uploader(
    "Upload an MRI Scan",
    type=["jpg", "jpeg", "png"],
    help="Upload a brain MRI image for analysis"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded MRI Scan", use_column_width=True)

    # Preprocess
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner("Analyzing MRI scan..."):
        predictions = model.predict(img_array)[0]

    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    with col2:
        st.markdown("### Analysis Result")

        if predicted_class == 'No Tumor':
            st.success(f"✅ {predicted_class}")
        else:
            st.error(f"⚠️ {predicted_class} Detected")

        st.metric("Confidence", f"{confidence:.1f}%")

        if confidence < 70:
            st.warning("⚠️ Low confidence prediction. Please consult a radiologist for confirmation.")
        else:
            st.info("Prediction confidence is reliable.")

        st.markdown("---")
        st.markdown("### About this finding")
        st.write(CLASS_INFO[predicted_class])

    # Probability breakdown
    st.markdown("---")
    st.markdown("### Probability Breakdown")
    for i, class_name in enumerate(CLASS_NAMES):
        prob = predictions[i] * 100
        st.progress(int(prob), text=f"{class_name}: {prob:.1f}%")

    st.markdown("---")
    st.caption("⚠️ NeuraSight is an AI assistance tool. Always consult a qualified medical professional for diagnosis.")