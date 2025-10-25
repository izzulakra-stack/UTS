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
        kelas = "🐱 Kucing"
    else:
        kelas = "🐶 Anjing"

    return kelas, lokasi, confidence

# ================================
# UI
# ================================
st.markdown("<h1 style='text-align:center;'>📷 Aplikasi Deteksi & Klasifikasi</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Pilih Mode: Hewan atau Mobil</h4>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Dikembangkan oleh: <b>Izzul Akrami</b></p>", unsafe_allow_html=True)
st.divider()

menu = st.sidebar.selectbox(
    "Pilih Mode:",
    ["Klasifikasi Hewan", "Deteksi Mobil (YOLO)"] if YOLO_AVAILABLE else ["Klasifikasi Hewan"]
)

uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

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
                st.success(f"✅ Gambar ini terdeteksi sebagai **{kelas}**")
                st.markdown(f"📍 Ditempatkan di: **{lokasi}**")
                st.write(f"Tingkat Kepercayaan: {confidence*100:.2f}%")
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
                        showroom_mapping = {
                            "Audi": "Showroom 1",
                            "Hyundai Creta": "Showroom 2",
                            "Mahindra Scorpio": "Showroom 3",
                            "Rolls Royce": "Showroom 4",
                            "Swift": "Showroom 5",
                            "Tata Safari": "Showroom 6",
                            "Toyota Innova": "Showroom 7"
                        }
                        showroom = showroom_mapping.get(label, "Showroom Tidak Diketahui")
                        st.success(f"✅ Mobil terdeteksi: **{label}**")
                        st.markdown(f"🏢 Ditempatkan di: **{showroom}**")
                else:
                    st.warning("🚫 Tidak ada mobil terdeteksi.")

                st.image(result_img, caption="🧾 Hasil Deteksi Mobil", use_container_width=True)
            except Exception as e:
                st.error(f"Gagal deteksi mobil: {e}")

else:
    st.info("Silakan unggah gambar terlebih dahulu untuk memulai prediksi.")
