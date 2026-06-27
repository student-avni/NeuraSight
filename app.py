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

# New Smart Summary with your "Confusion = Improvement" logic
def get_comparison_summary(class1, conf1, class2, conf2):
    summary_points = []

    # Point 1: Diagnosis Status
    if class1 == class2:
        summary_points.append(f"**Diagnosis Status:** Both scans show **{class1}**.")
    else:
        summary_points.append(f"**Diagnosis Change:** The AI detected a shift from **{class1}** to **{class2}**.")

    # Point 2: The "Confusion / Improvement" Check
    if class2 == 'No Tumor':
        summary_points.append("**Progress:** ✨ Fantastic news! No tumor was detected in the recent scan.")
    elif conf2 < 60:
        # If confidence is low, the AI is confused, which is a good sign for the patient!
        summary_points.append(f"**Progress:** The AI is confused by the second scan (only {conf2:.1f}% confident). This often means the tumor is shrinking, fading, or breaking apart! Please consult your doctor to confirm this improvement.")
    elif class1 == class2 and conf2 < (conf1 - 5):
        summary_points.append(f"**Progress:** AI confidence dropped by {abs(conf2 - conf1):.1f}%. This reduction usually indicates a positive response to treatment.")
    elif class1 == class2 and conf2 > (conf1 + 5):
        summary_points.append(f"**Progress:** AI confidence increased by {abs(conf2 - conf1):.1f}%.")
    
    # Point 3: Final Medical Conclusion
    if class1 != 'No Tumor' and class2 == 'No Tumor':
        summary_points.append("**Medical Conclusion:** ✅ Scan is clear. Excellent stability.")
    elif class1 == 'No Tumor' and class2 == 'No Tumor':
        summary_points.append("**Medical Conclusion:** ✅ Both scans are clear. Excellent stability.")
    else:
        summary_points.append("**Medical Conclusion:** 🏥 Please consult with your radiologist to verify these AI findings and compare the physical size.")

    return "\n\n".join([f"🔸 {item}" for item in summary_points])

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
    features = ["4-Class Tumor Detection", "MRI Scan Comparison", "Grad-CAM Heatmap", "PDF Reports"]
    for f in features:
        st.markdown(f"✅ {f}")
    st.markdown("---")
    st.markdown("### Model Info")
    st.markdown("**Architecture:** ResNet50")
    st.markdown("**Training Images:** 4,480")
    st.markdown("**Classes:** 4 tumor types")
    st.markdown("---")
    show_gradcam = st.toggle("Show Grad-CAM Heatmap", value=True)
    st.caption("⚠️ For research use only. Not a medical device.")

# Main header
st.markdown('<h1 class="main-header">🧠 NeuraSight</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Brain Tumor Detection & Analysis System</p>', unsafe_allow_html=True)

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

# TABS
tab1, tab2 = st.tabs(["🔍 Single Detection", "⚖️ Compare Scans"])

# ==========================================
# TAB 1: SINGLE DETECTION
# ==========================================
with tab1:
    uploaded_file = st.file_uploader(
        "Upload a single brain MRI image to detect tumors",
        type=["jpg", "jpeg", "png"],
        key="single_upload"
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
        
        # VALIDATION TRAP
        if confidence < 40:
            st.error("🚨 This may not be a valid brain MRI scan. Please upload a proper MRI image.")
        else:
            st.markdown("## 📊 Analysis Results")

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
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Original MRI Scan**")
                    st.image(image, use_container_width=True)
                with col2:
                    st.markdown("**Analysis Details**")
                    st.metric("Confidence", f"{confidence:.1f}%")
                    st.markdown(f"**About:** {info['desc']}")

            st.markdown("---")
            st.markdown("### 📄 Download Report")
            
            if st.button("Generate PDF Report", type="primary"):
                with st.spinner("Generating report..."):
                    import importlib
                    import report
                    importlib.reload(report)
                    from report import generate_pdf_report
                    heatmap_overlay = overlay_gradcam(image, heatmap) if heatmap is not None else None
                    
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

# ==========================================
# TAB 2: COMPARE SCANS
# ==========================================
with tab2:
    st.markdown("### ⚖️ Compare Two MRI Scans")
    st.info("ℹ️ Please upload two brain MRI scans of the same patient taken at different times for accurate comparison.")
    
    comp_col1, comp_col2 = st.columns(2)
    
    with comp_col1:
        scan1_file = st.file_uploader("Upload Scan 1 (Older)", type=["jpg", "jpeg", "png"], key="scan1")
    with comp_col2:
        scan2_file = st.file_uploader("Upload Scan 2 (Recent)", type=["jpg", "jpeg", "png"], key="scan2")
        
    if scan1_file and scan2_file:
        img1 = Image.open(scan1_file).convert('RGB')
        img2 = Image.open(scan2_file).convert('RGB')
        
        img_arr1 = np.expand_dims(np.array(img1.resize((224, 224))) / 255.0, axis=0)
        img_arr2 = np.expand_dims(np.array(img2.resize((224, 224))) / 255.0, axis=0)
        
        with st.spinner("🔍 Comparing scans..."):
            pred1 = model.predict(img_arr1)[0]
            pred2 = model.predict(img_arr2)[0]
            
            heat1 = generate_gradcam(model, img_arr1) if show_gradcam else None
            heat2 = generate_gradcam(model, img_arr2) if show_gradcam else None
            
        class1 = CLASS_NAMES[np.argmax(pred1)]
        conf1 = np.max(pred1) * 100
        class2 = CLASS_NAMES[np.argmax(pred2)]
        conf2 = np.max(pred2) * 100
        
        st.markdown("---")
        
        # VALIDATION TRAP (If either image is really bad, stop!)
        if conf1 < 40 or conf2 < 40:
            st.error("🚨 This may not be a valid brain MRI scan. Please upload a proper MRI image.")
        else:
            # Everything is good, show the results!
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.markdown(f"### Scan 1: {class1} ({conf1:.1f}%)")
                if heat1 is not None:
                    st.image(overlay_gradcam(img1, heat1), use_container_width=True)
                else:
                    st.image(img1, use_container_width=True)
                    
            with res_col2:
                st.markdown(f"### Scan 2: {class2} ({conf2:.1f}%)")
                if heat2 is not None:
                    st.image(overlay_gradcam(img2, heat2), use_container_width=True)
                else:
                    st.image(img2, use_container_width=True)

            # Smart Bullet Point Summary Box
            st.markdown("---")
            st.markdown("### 📋 Comparison Summary")
            summary_text = get_comparison_summary(class1, conf1, class2, conf2)
            st.info(summary_text)