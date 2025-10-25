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
        kelas = "Anjing"
        ikon = "🐶"
    else:
        kelas = "Kucing"
        ikon = "🐱"

    return kelas, ikon, confidence

# ================================
# Fungsi menggambar label di gambar
# ================================
def draw_label(img_cv, label, lokasi, bbox=None):
    color = (0, 255, 0)  # Hijau
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    text = f"{label} → {lokasi}"

    if bbox is not None:
        # bbox = [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_cv, text, (x1, y1-10), font, font_scale, color, thickness, cv2.LINE_AA)
    else:
        # untuk hewan, taruh di pojok atas
        cv2.putText(img_cv, text, (10, 30), font, font_scale, color, thickness, cv2.LINE_AA)

    return img_cv

# ================================
# UI
# ================================
st.markdown("<h1 style='text-align:center;'>📷 Aplikasi Deteksi & Klasifikasi Interaktif</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Mobil ke Showroom & Hewan ke Kandang</h4>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Dikembangkan oleh: <b>Izzul Akrami</b> | UTS Big Data 2025</p>", unsafe_allow_html=True)
st.divider()

uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📸 Gambar yang Diupload", use_container_width=True)

    img_cv = np.array(img)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    hasil_lokasi = []

    # ---- Klasifikasi Hewan ----
    if classifier is not None:
        try:
            kelas, ikon, confidence = klasifikasi_hewan(img, classifier)
            if kelas == "Anjing":
                lokasi = "Kandang Anjing 🐶"
            else:
                lokasi = "Kandang Kucing 🐱"

            hasil_lokasi.append(f"{ikon} {kelas} → {lokasi} (Confidence: {confidence*100:.2f}%)")
            img_cv = draw_label(img_cv, kelas, lokasi)
        except Exception as e:
            st.error(f"Gagal klasifikasi hewan: {e}")

    # ---- Deteksi Mobil YOLO ----
    if YOLO_AVAILABLE and yolo_model is not None:
        try:
            results = yolo_model(img)
            if len(results[0].boxes) > 0:
                for i, box in enumerate(results[0].boxes):
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    label = results[0].names[int(box.cls[0])]
                    showroom_mapping = {
                        "Audi": "Showroom 1 🚗",
                        "Hyundai Creta": "Showroom 2 🚗",
                        "Mahindra Scorpio": "Showroom 3 🚗",
                        "Rolls Royce": "Showroom 4 🚗",
                        "Swift": "Showroom 5 🚗",
                        "Tata Safari": "Showroom 6 🚗",
                        "Toyota Innova": "Showroom 7 🚗"
                    }
                    lokasi = showroom_mapping.get(label, "Showroom Tidak Diketahui")
                    hasil_lokasi.append(f"{label} → {lokasi}")
                    img_cv = draw_label(img_cv, label, lokasi, bbox=[x1, y1, x2, y2])
            else:
                st.warning("🚫 Tidak ada mobil terdeteksi.")
        except Exception as e:
            st.error(f"Gagal deteksi mobil: {e}")

    # ---- Tampilkan Semua Hasil Lokasi ----
    if hasil_lokasi:
        st.markdown("### 📍 Hasil Penempatan Objek:")
        for item in hasil_lokasi:
            st.write(f"- {item}")

    # Tampilkan gambar hasil deteksi + label
    img_cv_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    st.image(img_cv_rgb, caption="🧾 Hasil Deteksi & Penempatan Objek", use_container_width=True)

else:
    st.info("Silakan unggah gambar terlebih dahulu untuk memulai prediksi.")
