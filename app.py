import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Page config - must be first
st.set_page_config(
    page_title="NeuraSight | AI Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        border: 1px solid #444;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .result-box {
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .disclaimer {
        background: #1a1a2e;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.85rem;
        color: #888;
        margin-top: 2rem;
    }
    .feature-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.2rem;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1e2e 0%, #16213e 100%);
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('tumor_model.h5')
    return model

model = load_model()

CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

CLASS_INFO = {
    'Glioma': {
        'desc': 'A tumor originating in the glial cells of the brain or spine.',
        'severity': 'High',
        'color': '#ff4444',
        'action': 'Immediate neurological consultation required.',
        'symptoms': 'Headaches, seizures, memory loss, personality changes'
    },
    'Meningioma': {
        'desc': 'A tumor forming on the membranes surrounding the brain and spinal cord.',
        'severity': 'Medium',
        'color': '#ff8800',
        'action': 'Schedule appointment with neurosurgeon within 1-2 weeks.',
        'symptoms': 'Headaches, vision problems, hearing loss, memory difficulties'
    },
    'No Tumor': {
        'desc': 'No signs of tumor detected in the MRI scan.',
        'severity': 'None',
        'color': '#00cc44',
        'action': 'Continue regular health checkups as scheduled.',
        'symptoms': 'N/A'
    },
    'Pituitary': {
        'desc': 'A tumor developing in the pituitary gland at the base of the brain.',
        'severity': 'Low-Medium',
        'color': '#ffaa00',
        'action': 'Endocrinology consultation recommended.',
        'symptoms': 'Vision changes, hormonal imbalances, headaches'
    }
}

def generate_gradcam(model, img_array, layer_name='conv5_block3_out'):
    try:
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    except:
        return None

def overlay_gradcam(original_img, heatmap):
    img = np.array(original_img.resize((224, 224)))
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    overlay = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
    return overlay

# Sidebar
with st.sidebar:
    st.markdown("## 🧠 NeuraSight")
    st.markdown("---")
    st.markdown("### About")
    st.info("NeuraSight uses deep learning to analyze brain MRI scans and detect tumor types with AI-powered explanations.")
    st.markdown("### Features")
    features = ["4-Class Tumor Detection", "Grad-CAM Heatmap", "Confidence Scoring", "Severity Assessment", "Clinical Recommendations"]
    for f in features:
        st.markdown(f"✅ {f}")
    st.markdown("---")
    st.markdown("### Model Info")
    st.markdown("**Architecture:** ResNet50")
    st.markdown("**Training Images:** 4,480")
    st.markdown("**Classes:** 4 tumor types")
    st.markdown("**Framework:** TensorFlow/Keras")
    st.markdown("---")
    show_gradcam = st.toggle("Show Grad-CAM Heatmap", value=True)
    st.markdown("---")
    st.caption("⚠️ For research use only. Not a medical device.")

# Main header
st.markdown('<h1 class="main-header">🧠 NeuraSight</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Brain Tumor Detection & Analysis System</p>', unsafe_allow_html=True)

# Badges
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.markdown("""
    <div style="text-align:center">
        <span class="feature-badge">🔬 ResNet50</span>
        <span class="feature-badge">🎯 Grad-CAM</span>
        <span class="feature-badge">📊 4-Class Detection</span>
        <span class="feature-badge">📄 PDF Report</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Upload section
# Patient Details Form
st.markdown("### 👤 Patient Information")
col1, col2, col3 = st.columns(3)
with col1:
    patient_name = st.text_input("Patient Name", placeholder="Enter patient name")
with col2:
    patient_age = st.number_input("Age", min_value=1, max_value=120, value=25)
with col3:
    doctor_name = st.text_input("Doctor Name", placeholder="Enter doctor name")

st.markdown("---")
uploaded_file = st.file_uploader(
    "Drag and drop or click to upload",
    type=["jpg", "jpeg", "png"],
    help="Upload a brain MRI image in JPG or PNG format"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("🔍 Analyzing MRI scan with AI..."):
        predictions = model.predict(img_array)[0]
        heatmap = generate_gradcam(model, img_array) if show_gradcam else None

    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    info = CLASS_INFO[predicted_class]

    st.markdown("---")
    st.markdown("## 📊 Analysis Results")

    # Top metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("🎯 Detected", predicted_class)
    with m2:
        st.metric("📊 Confidence", f"{confidence:.1f}%")
    with m3:
        st.metric("⚠️ Severity", info['severity'])
    with m4:
        reliability = "High" if confidence >= 80 else "Medium" if confidence >= 60 else "Low"
        st.metric("🔍 Reliability", reliability)

    st.markdown("---")

    # Images side by side
    if heatmap is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Original MRI Scan**")
            st.image(image, use_container_width=True)
        with col2:
            st.markdown("**Grad-CAM Heatmap**")
            overlay = overlay_gradcam(image, heatmap)
            st.image(overlay, use_container_width=True)
        with col3:
            st.markdown("**Analysis Details**")
            if predicted_class == 'No Tumor':
                st.success(f"✅ {predicted_class}")
            elif info['severity'] == 'High':
                st.error(f"🚨 {predicted_class} Detected")
            else:
                st.warning(f"⚠️ {predicted_class} Detected")

            if confidence < 60:
                st.error("🔴 Low Confidence - Consult radiologist")
            elif confidence < 80:
                st.warning("🟡 Medium Confidence")
            else:
                st.success("🟢 High Confidence")

            st.markdown(f"**About:** {info['desc']}")
            st.markdown(f"**Action:** {info['action']}")
            st.markdown(f"**Symptoms:** {info['symptoms']}")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original MRI Scan**")
            st.image(image, use_container_width=True)
        with col2:
            st.markdown("**Analysis Details**")
            if predicted_class == 'No Tumor':
                st.success(f"✅ {predicted_class}")
            else:
                st.error(f"⚠️ {predicted_class} Detected")
            st.metric("Confidence", f"{confidence:.1f}%")
            st.markdown(f"**About:** {info['desc']}")
            st.markdown(f"**Recommended Action:** {info['action']}")

    st.markdown("---")

    # Probability breakdown
    st.markdown("### 📈 Probability Breakdown")
    cols = st.columns(4)
    for i, (class_name, col) in enumerate(zip(CLASS_NAMES, cols)):
        prob = predictions[i] * 100
        with col:
            st.metric(class_name, f"{prob:.1f}%")
            st.progress(int(prob))

    st.markdown("---")
    # PDF Report download
    st.markdown("---")
    st.markdown("### 📄 Download Report")
    
    if st.button("Generate PDF Report", type="primary"):
        with st.spinner("Generating report..."):
            import importlib
            import report
            importlib.reload(report)
            from report import generate_pdf_report
            heatmap_overlay = overlay_gradcam(image, heatmap) if heatmap is not None else None
            
            # Here is the fix: we added patient_name, patient_age, and doctor_name
            pdf_path = generate_pdf_report(
                image, predicted_class, confidence,
                predictions, CLASS_NAMES, heatmap_overlay,
                patient_name, patient_age, doctor_name
            )
            
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name="NeuraSight_Report.pdf",
                mime="application/pdf"
            )
            st.success("Report generated!")

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
    ⚠️ <strong>Medical Disclaimer:</strong> NeuraSight is an AI-powered research tool and is NOT a certified medical device.
    Results should not be used as a substitute for professional medical diagnosis.
    Always consult a qualified neurologist or radiologist for medical decisions.
    </div>
    """, unsafe_allow_html=True)

else:
    # Landing state
    st.markdown("### 👆 Upload an MRI scan to begin analysis")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🔬 **Step 1**\nUpload a brain MRI scan in JPG or PNG format")
    with col2:
        st.info("🤖 **Step 2**\nAI analyzes the scan using deep learning")
    with col3:
        st.info("📊 **Step 3**\nGet detailed results with heatmap visualization")