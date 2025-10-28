import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
from streamlit_lottie import st_lottie
import requests

# 🧩 Fix macOS TensorFlow thread & mutex issues
# (This part is essential for stability on some systems, so we keep it)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ------------------------- #
# ⚙️ Page Configuration
# ------------------------- #
# Changed title, icon, and layout from "wide" to "centered"
st.set_page_config(page_title="AI X-Ray Analyzer", page_icon="🤖", layout="centered")

# ------------------------- #
# 🎨 Custom Styling (Completely New Theme)
# ------------------------- #
st.markdown("""
<style>
/* Background */
.main {
    background-image: url("https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&q=80&w=2070"); /* New elegant medical background */
    background-size: cover; /* Cover the entire page */
    background-repeat: no-repeat; /* Don't repeat the image */
    background-attachment: fixed; /* Fix the background on scroll */
    color: #e2e8f0; /* Light gray text */
    font-family: 'Poppins', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(30, 41, 59, 0.9); /* Darker slate with 90% opacity */
    backdrop-filter: blur(5px); /* Blur the background behind the sidebar */
    border-right: 2px solid #334155;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #10b981, #0d9488); /* Green gradient */
    color: white;
    border: none;
    border-radius: 10px;
    font-size: 17px;
    padding: 10px 30px;
    transition: all 0.3s ease-in-out;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2);
}
.stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #0f9d6d, #0a7a6a);
}

/* Animated Title */
@keyframes gradientShift {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

h1 {
  background: linear-gradient(270deg, #10b981, #2dd4bf, #67e8f9); /* New gradient */
  background-size: 600% 600%;
  animation: gradientShift 6s ease infinite;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  text-align: center;
  font-weight: 900;
  font-size: 44px; /* Slightly smaller */
}

h3 {
  text-align: center;
  font-weight: 400;
  color: #94a3b8; /* Lighter slate color */
  margin-bottom: 30px;
}

/* Progress bar */
.stProgress > div > div > div > div {
    background-color: #10b981 !important; /* Match green theme */
}

/* Remove default Streamlit footer */
footer {visibility: hidden;}

/* Style the file uploader */
.stFileUploader {
    border: 2px dashed #334155;
    border-radius: 15px;
    background-color: rgba(30, 41, 59, 0.8); /* Slate with 80% opacity */
    backdrop-filter: blur(5px); /* Blur the background behind it */
    padding: 25px;
    text-align: center;
}

/* Style success/error boxes (replaces .result-card) */
[data-testid="stAlert"] {
    border-radius: 10px;
    font-size: 16px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ------------------------- #
# 🧠 Sidebar Info (Rewritten)
# ------------------------- #
# Using a new icon
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2179/2179603.png", width=80)
st.sidebar.title("🤖 How This Works")
st.sidebar.info("""
This app uses a **Convolutional Neural Network (CNN)**, a type of Deep Learning, to analyze chest X-ray images.

The model was trained on a large dataset to identify patterns associated with **Pneumonia**.

**Disclaimer:** This is an educational tool, not a substitute for professional medical diagnosis.
""")

# ------------------------- #
# 📦 Load Model
# ------------------------- #
@st.cache_resource
def load_model():
    model_path = "pneumonia_model_fixed.keras" # User must have this file
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        st.info("Please make sure the 'pneumonia_model_fixed.keras' file is in the same directory as the app.")
        st.stop()
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        # st.success("✅ Model loaded successfully!") # Hiding this for a cleaner UI
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()
    return model

model = load_model()

# ------------------------- #
# 🌈 Animated Title (Rewritten)
# ------------------------- #
st.markdown("<h1>🤖 Chest X-Ray AI Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<h3>AI-Powered Pneumonia Screening</h3>", unsafe_allow_html=True)

# ------------------------- #
# 📤 Upload Section
# ------------------------- #
# Replaced the custom HTML box with a standard (but styled) file uploader
uploaded_file = st.file_uploader("Upload your X-ray image (JPG, PNG)", type=["jpg", "jpeg", "png"])

# ------------------------- #
# 🔄 Image Preprocessing
# ------------------------- #
# Renamed function from preprocess_image to prepare_image
def prepare_image(img):
    img = img.resize((64, 64)) # Must match the model's input size
    img_array = np.array(img)
    if img_array.shape[-1] == 4: # Handle RGBA images
        img_array = img_array[..., :3]
    img_array = img_array.astype("float32") / 255.0
    return np.expand_dims(img_array, axis=0) # Add batch dimension

# ------------------------- #
# 💫 Lottie Animation Loader
# ------------------------- #
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Using a new Lottie animation (AI brain scan)
lottie_brain = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_A8yN3W.json")

# ------------------------- #
# 🔍 Prediction Section (New Layout)
# ------------------------- #
# Changed from 2-column layout to a single sequential layout
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    
    # 1. Show the image first
    st.image(img, caption="Uploaded X-ray for Analysis", use_container_width=True)
    
    # 2. Add a spinner for the whole analysis process
    with st.spinner("AI is analyzing the image... Please wait."):
        
        # 3. Show Lottie animation *inside* the spinner
        if lottie_brain:
            st_lottie(lottie_brain, height=150, key="analyze")
        
        # 4. Preprocess
        img_array = prepare_image(img)
        
        # 5. Predict
        prediction = model.predict(img_array)
        confidence = float(prediction[0][0])
        time.sleep(1.5) # Keep delay for dramatic effect

    # 6. Show results *after* spinner
    st.markdown("---") # Add a separator
    st.markdown("## 🔬 Analysis Complete")

    # Using st.error and st.success instead of custom HTML cards
    if confidence > 0.5:
        prob = confidence * 100
        st.error(f"**Result:** ⚠️ Potential Pneumonia Signs Found\n\n**Confidence:** {prob:.2f}%")
        st.warning("**Disclaimer:** Please consult a medical professional immediately for a proper diagnosis.")
    else:
        prob = (1 - confidence) * 100
        st.success(f"**Result:** ✅ Signs Point to Normal\n\n**Confidence:** {prob:.2f}%")
        st.info("**Disclaimer:** This AI screening did not detect signs of pneumonia. Always consult a professional for medical advice.")
        st.balloons()

# ------------------------- #
# ⚙️ Footer (Rewritten)
# ------------------------- #
st.markdown("---")
st.markdown("<center>🤖 AI Analyzer built with Streamlit & TensorFlow</center>", unsafe_allow_html=True)

