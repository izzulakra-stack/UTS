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
# Page Config
# ================================
st.set_page_config(
    page_title="📷 Aplikasi Deteksi & Klasifikasi",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# Styling Dashboard
# ================================
st.markdown("""
    <style>
    /* Warna dasar */
    .stApp {
        background-color: #e3f2fd;
        color: #0b3d91;
    }

    /* Header utama */
    .main-header {
        background: linear-gradient(135deg, #64b5f6 0%, #bbdefb 100%);
        padding: 35px 25px;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.15);
        margin-bottom: 35px;
    }

    .main-header h1 {
        margin-bottom: 10px;
        font-size: 2.8em;
        font-weight: 800;
        letter-spacing: 1px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }

    .main-header p {
        font-size: 1.2em;
        margin: 3px 0;
    }

    .main-header .npm {
        font-size: 1em;
        color: #e3f2fd;
        margin-top: 5px;
    }

    .main-header .subtext {
        font-size: 1em;
        color: #e3f2fd;
        font-style: italic;
        margin-top: 2px;
    }

    /* Tombol */
    .stButton>button {
        background-color: #0b77d6; 
        color: white;
        border-radius: 10px;
        padding: 0.6em 1.2em;
        border: none;
    }
    .stButton>button:hover {
        background-color: #095cb5;
    }

    /* Sidebar */
    .stSidebar {
        background-color: #bbdefb;
    }

    /* Kotak hasil */
    .kotak-hewan {
        background-color: #b3e5fc; 
        padding: 15px; 
        border-radius: 12px; 
        margin-bottom: 10px;
    }

    .kotak-mobil {
        background-color: #c8e6c9; 
        padding: 15px; 
        border-radius: 12px; 
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ================================
# UI Header
# ================================
st.markdown("""
<div class="main-header">
    <h1>📷 APLIKASI DETEKSI & KLASIFIKASI GAMBAR</h1>
    <p><b>Izzul Akrami</b></p>
    <p class="npm">NPM: 2208108010026</p>
    <p class="subtext">Jurusan Statistika — Universitas Syiah Kuala</p>
</div>
""", unsafe_allow_html=True)

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
                input_shape = classifier.input_shape[1:3]
                img_resized = img.resize(input_shape)
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                prediction = classifier.predict(img_array)
                class_index = int(prediction[0][0] > 0.5)
                kelas = label_mapping.get(class_index, "Unknown")
                confidence = float(np.max(prediction))
                lokasi_mapping = {"Cat": "Kandang Kucing", "Dog": "Kandang Anjing"}
                lokasi = lokasi_mapping.get(kelas, "Kandang Tidak Diketahui")
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
        with st.spinner("🚗 Sedang mendeteksi objek..."):
            try:
                results = yolo_model(img, conf=0.5)
                result_img = results[0].plot()

                detected_objects = []
                for box in results[0].boxes:
                    label = results[0].names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    if label in ["car", "truck"]:
                        detected_objects.append((label.capitalize(), conf))

                if detected_objects:
                    for obj, conf in detected_objects:
                        st.markdown(f"""
                        <div class='kotak-mobil'>
                            <h3>✅ {obj} terdeteksi!</h3>
                            <p>📊 Confidence: {conf*100:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("🚫 Tidak ada mobil atau truk terdeteksi.")
                st.image(result_img, caption="🧾 Hasil Deteksi", use_container_width=True)
            except Exception as e:
                st.error(f"Gagal deteksi objek: {e}")
    else:
        st.info("Silakan unggah gambar terlebih dahulu untuk memulai deteksi. 📂")
