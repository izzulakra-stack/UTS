import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
from ultralytics import YOLO
import cv2

YOLO_AVAILABLE = True

# ================================
# Load Models
# ================================
@st.cache_resource
def load_models():
    yolo_model, classifier = None, None
    try:
        if YOLO_AVAILABLE:
            yolo_model = YOLO("Model/deteksi.pt")
        classifier = tf.keras.models.load_model("Model/klasifikasi.h5")
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ================================
# Fungsi Klasifikasi Hewan
# ================================
def klasifikasi_hewan(img, model):
    input_shape = model.input_shape[1:3]
    img_resized = img.resize(input_shape)
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    if class_index == 0:
        kelas = "🐶 Anjing"
        lokasi = "Kandang Anjing"
    else:
        kelas = "🐱 Kucing"
        lokasi = "Kandang Kucing"

    return kelas, lokasi, confidence

# ================================
# Styling Dashboard
# ================================
st.set_page_config(
    page_title="📷 Aplikasi Deteksi & Klasifikasi",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    body {
        background-color: #f0f8ff;
        color: #1a1a1a;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stSidebar {
        background-color: #e6f2ff;
    }
    .stAlert {
        background-color: #ffebcc;
    }
    .kotak-hewan {
        background-color: #cce5ff;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .kotak-mobil {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ================================
# UI
# ================================
st.markdown("<h1 style='text-align:center; color:#0b3d91;'>📷 Aplikasi Deteksi & Klasifikasi</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:#0b3d91;'>Pilih Mode: Hewan atau Mobil</h4>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#0b3d91;'>Dikembangkan oleh: <b>Izzul Akrami</b></p>", unsafe_allow_html=True)
st.divider()

menu = st.sidebar.selectbox(
    "Pilih Mode 🖥️:",
    ["Klasifikasi Hewan", "Deteksi Mobil (YOLO)"] if YOLO_AVAILABLE else ["Klasifikasi Hewan"]
)

uploaded_file = st.file_uploader("Unggah Gambar 📤", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="📸 Gambar yang Diupload", use_container_width=True)

    # ================================
    # Mode Klasifikasi Hewan
    # ================================
    if menu == "Klasifikasi Hewan" and classifier is not None:
        with st.spinner("🔍 Sedang mengklasifikasi..."):
            try:
                kelas, lokasi, confidence = klasifikasi_hewan(img, classifier)
                st.markdown(f"""
                <div class='kotak-hewan'>
                    <h3>✅ {kelas}</h3>
                    <p>📍 {lokasi}</p>
                    <p>📊 Confidence: {confidence*100:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Gagal melakukan klasifikasi: {e}")

    # ================================
    # Mode Deteksi Mobil
    # ================================
    elif menu == "Deteksi Mobil (YOLO)" and YOLO_AVAILABLE and yolo_model is not None:
        with st.spinner("🚗 Sedang mendeteksi mobil..."):
            try:
                results = yolo_model(img)
                result_img = results[0].plot()

                if len(results[0].boxes) > 0:
                    for i, box in enumerate(results[0].boxes):
                        label = results[0].names[int(box.cls[0])]
                        st.markdown(f"""
                        <div class='kotak-mobil'>
                            <h3>✅ Mobil terdeteksi: {label}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("🚫 Tidak ada mobil terdeteksi.")

                st.image(result_img, caption="🧾 Hasil Deteksi Mobil", use_container_width=True)
            except Exception as e:
                st.error(f"Gagal deteksi mobil: {e}")

else:
    st.info("Silakan unggah gambar terlebih dahulu untuk memulai prediksi. 📂")
