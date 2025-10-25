import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import subprocess
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
            yolo_model = YOLO("Model/deteksi.pt")  # model YOLO
        classifier = tf.keras.models.load_model("Model/klasifikasi.h5")  # model klasifikasi
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ================================
# UI
# ================================
st.markdown("<h1 style='text-align: center;'>🐐 SmartVision</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Deteksi Objek dan Klasifikasi Gambar Otomatis</h3>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Dibuat oleh: <b>Izzul Akrami</b> | UTS Big Data 2025</p>", unsafe_allow_html=True)


menu = st.sidebar.selectbox(
    "Pilih Mode:",
    ["Klasifikasi Gambar", "Deteksi Objek (YOLO)"] if YOLO_AVAILABLE else ["Klasifikasi Gambar"]
)

uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="📸 Gambar yang Diupload", use_container_width=True)

    # ================================
    # Mode YOLO (jika tersedia)
    # ================================
    if menu == "Deteksi Objek (YOLO)" and YOLO_AVAILABLE and yolo_model is not None:
        with st.spinner("🔍 Sedang mendeteksi objek..."):
            results = yolo_model(img)
            result_img = results[0].plot()
            st.image(result_img, caption="🧾 Hasil Deteksi", use_container_width=True)

    # ================================
    # Mode Klasifikasi (auto resize)
    # ================================
    elif menu == "Klasifikasi Gambar" and classifier is not None:
        with st.spinner("🧩 Sedang mengklasifikasi..."):
            # Ambil ukuran input model otomatis
            try:
                input_shape = classifier.input_shape[1:3]  # contoh (128, 128)
                st.write(f"📏 Ukuran input model: {input_shape}")
                img_resized = img.resize(input_shape)
            except Exception:
                st.warning("⚠️ Tidak bisa mendeteksi ukuran input model, pakai 224x224 default.")
                img_resized = img.resize((224, 224))

            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Prediksi
            try:
                prediction = classifier.predict(img_array)
                class_index = np.argmax(prediction)
                confidence = np.max(prediction)

                st.success("✅ Hasil Prediksi:")
                st.write("**Kelas (Index):**", class_index)
                st.write("**Tingkat Kepercayaan:**", f"{confidence*100:.2f}%")
            except Exception as e:
                st.error(f"Gagal melakukan prediksi: {e}")

    else:
        st.warning("⚠️ Model belum dimuat atau mode tidak tersedia.")
else:
    st.info("Silakan unggah gambar terlebih dahulu untuk memulai prediksi.")
