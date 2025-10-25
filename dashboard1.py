import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
from ultralytics import YOLO

YOLO_AVAILABLE = True

# ================================
# Load Models
# ================================
@st.cache_resource
def load_models():
    yolo_model, classifier, label_mapping = None, None, {}
    try:
        if YOLO_AVAILABLE:
            yolo_model = YOLO("Model/deteksi.pt")
        classifier = tf.keras.models.load_model("Model/klasifikasi.h5")
        label_mapping = {0: "Cat", 1: "Dog"}
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
    return yolo_model, classifier, label_mapping

yolo_model, classifier, label_mapping = load_models()

# ================================
# Fungsi Klasifikasi Hewan
# ================================
def klasifikasi_hewan(img, model, label_mapping):
    input_shape = model.input_shape[1:3]
    img_resized = img.resize(input_shape)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    if prediction.shape[1] == 1:
        class_index = int(prediction[0][0] > 0.5)
    else:
        class_index = np.argmax(prediction[0])

    confidence = np.max(prediction)
    kelas = label_mapping.get(class_index, "Unknown")
    lokasi_mapping = {"Cat": "Kandang Kucing", "Dog": "Kandang Anjing"}
    lokasi = lokasi_mapping.get(kelas, "Kandang Tidak Diketahui")

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
    body {background-color: #f0f8ff; color: #1a1a1a;}
    .stButton>button {background-color: #4CAF50; color: white;}
    .stSidebar {background-color: #e6f2ff;}
    .kotak-hewan {background-color: #cce5ff; padding: 15px; border-radius: 10px; margin-bottom: 10px;}
    .kotak-mobil {background-color: #d4edda; padding: 15px; border-radius: 10px; margin-bottom: 10px;}
    </style>
""", unsafe_allow_html=True)

# ================================
# UI
# ================================
st.markdown("<h1 style='text-align:center; color:#0b3d91;'>📷 Aplikasi Deteksi & Klasifikasi</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#0b3d91;'>Dikembangkan oleh: <b>Izzul Akrami</b></p>", unsafe_allow_html=True)
st.divider()

# Sidebar Menu
menu = st.sidebar.selectbox(
    "Pilih Halaman 🖥️:",
    ["🏠 Home", "Klasifikasi Hewan", "Deteksi Mobil (YOLO)"] if YOLO_AVAILABLE else ["🏠 Home", "Klasifikasi Hewan"]
)

# ================================
# HALAMAN HOME
# ================================
if menu == "🏠 Home":
    st.markdown("""
    <div style='text-align:center;'>
        <h2>Selamat Datang di Aplikasi Deteksi & Klasifikasi Gambar</h2>
        <p>Aplikasi ini dapat mengenali dua jenis objek:</p>
        <ul style='text-align:left; display:inline-block; text-align:justify;'>
            <li>🐱 <b>Klasifikasi Hewan:</b> Membedakan antara <i>Kucing</i> dan <i>Anjing</i>.</li>
            <li>🚗 <b>Deteksi Mobil (YOLO):</b> Mendeteksi keberadaan mobil di dalam gambar.</li>
        </ul>
        <p>Gunakan menu di sebelah kiri untuk memilih mode yang diinginkan.<br>
        Pastikan Anda mengunggah gambar dengan format <b>JPG</b>, <b>JPEG</b>, atau <b>PNG</b>.</p>
    </div>
    """, unsafe_allow_html=True)

   # Gambar ilustrasi di tengah
st.header("Contoh Gambar")
col1, col2, col3 = st.columns(3)

with col1:
    st.image("https://images.unsplash.com/photo-1601758064223-6f3be8b1b5e3?auto=format&fit=crop&w=500&q=80", 
             caption="Anjing", use_column_width=True)

with col2:
    st.image("https://images.unsplash.com/photo-1518791841217-8f162f1e1131?auto=format&fit=crop&w=500&q=80", 
             caption="Kucing", use_column_width=True)

with col3:
    st.image("https://images.unsplash.com/photo-1502877338535-766e1452684a?auto=format&fit=crop&w=500&q=80", 
             caption="Mobil", use_column_width=True)

# ================================
# HALAMAN KLASIFIKASI HEWAN
# ================================
elif menu == "Klasifikasi Hewan" and classifier is not None:
    uploaded_file = st.file_uploader("Unggah Gambar 📤", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="📸 Gambar yang Diupload", use_container_width=True)
        with st.spinner("🔍 Sedang mengklasifikasi..."):
            try:
                kelas, lokasi, confidence = klasifikasi_hewan(img, classifier, label_mapping)
                st.markdown(f"""
                <div class='kotak-hewan'>
                    <h3>✅ {kelas}</h3>
                    <p>📍 {lokasi}</p>
                    <p>📊 Confidence: {confidence*100:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Gagal melakukan klasifikasi: {e}")
    else:
        st.info("Silakan unggah gambar terlebih dahulu untuk memulai klasifikasi. 📂")

# ================================
# HALAMAN DETEKSI MOBIL (YOLO)
# ================================
elif menu == "Deteksi Mobil (YOLO)" and YOLO_AVAILABLE and yolo_model is not None:
    uploaded_file = st.file_uploader("Unggah Gambar 📤", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="📸 Gambar yang Diupload", use_container_width=True)
        with st.spinner("🚗 Sedang mendeteksi mobil..."):
            try:
                results = yolo_model(img)
                result_img = results[0].plot()

                detected_labels = set()
                for box in results[0].boxes:
                    label = results[0].names[int(box.cls[0])]
                    if label == "car":
                        detected_labels.add(label)

                if detected_labels:
                    for label in detected_labels:
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
        st.info("Silakan unggah gambar terlebih dahulu untuk memulai deteksi. 📂")
