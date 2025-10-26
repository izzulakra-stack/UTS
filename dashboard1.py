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
    page_title="🤖 Aplikasi Deteksi & Klasifikasi Gambar",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# Styling Futuristic
# ================================
st.markdown("""
<style>
/* Latar belakang utama */
.stApp {
    background: radial-gradient(circle at 20% 20%, #0a0f24 0%, #000000 100%);
    color: #00e5ff;
    font-family: 'Consolas', 'Courier New', monospace;
}

/* Header */
.main-header {
    background: linear-gradient(135deg, rgba(0, 255, 255, 0.15), rgba(0, 0, 0, 0.7));
    border: 2px solid rgba(0, 255, 255, 0.4);
    border-radius: 20px;
    padding: 30px 25px;
    margin-bottom: 35px;
    text-align: center;
    box-shadow: 0 0 25px rgba(0, 255, 255, 0.4);
}

.main-header h1 {
    color: #00ffff;
    font-size: 2.8em;
    font-weight: 900;
    text-shadow: 0 0 10px #00e5ff, 0 0 20px #00e5ff;
    letter-spacing: 2px;
    margin-bottom: 10px;
}

.main-header p {
    margin: 5px 0;
    font-size: 1.1em;
    color: #b2ebf2;
}

.main-header .npm {
    color: #80deea;
    font-size: 1em;
}

.main-header .subtext {
    font-style: italic;
    color: #4dd0e1;
}

/* Tombol */
.stButton>button {
    background-color: #00bcd4;
    border: none;
    color: black;
    font-weight: bold;
    border-radius: 8px;
    padding: 0.6em 1.2em;
    box-shadow: 0 0 10px #00e5ff;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background-color: #00e5ff;
    color: black;
    box-shadow: 0 0 20px #00ffff;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(0, 20, 30, 0.95);
    border-right: 2px solid #00e5ff;
    box-shadow: 0 0 15px rgba(0, 255, 255, 0.2);
}

/* Kotak hasil */
.result-box {
    background: rgba(0, 255, 255, 0.1);
    border: 1px solid #00e5ff;
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 0 10px #00e5ff;
    margin-top: 15px;
}
.result-box h3 {
    color: #00ffff;
    text-shadow: 0 0 10px #00ffff;
}

/* Judul subhalaman */
h2 {
    color: #00e5ff;
    text-shadow: 0 0 8px #00e5ff;
}

/* Link dan highlight */
a, .highlight {
    color: #00ffff;
}
</style>
""", unsafe_allow_html=True)

# ================================
# Header UI
# ================================
st.markdown("""
<div class="main-header">
    <h1>🤖 APLIKASI DETEKSI & KLASIFIKASI GAMBAR</h1>
    <p><b>Izzul Akrami</b></p>
    <p class="npm">NPM: 2208108010026</p>
    <p class="subtext">Jurusan Statistika — Universitas Syiah Kuala</p>
</div>
""", unsafe_allow_html=True)

menu = st.sidebar.selectbox(
    "🧭 Navigasi Menu",
    ["🏠 Home", "Klasifikasi Hewan", "Deteksi Mobil (YOLO)"] if YOLO_AVAILABLE else ["🏠 Home", "Klasifikasi Hewan"]
)

# ================================
# Halaman HOME
# ================================
if menu == "🏠 Home":
    st.markdown("""
    <div style='text-align:center;'>
        <h2>Selamat Datang di Sistem Cerdas Deteksi & Klasifikasi</h2>
        <p>🧠 Aplikasi ini dirancang untuk mengenali dua jenis objek dengan teknologi kecerdasan buatan:</p>
        <ul style='text-align:left; display:inline-block;'>
            <li>🐱 <b>Klasifikasi Hewan:</b> Membedakan antara Kucing dan Anjing.</li>
            <li>🚗 <b>Deteksi Mobil (YOLO):</b> Mengidentifikasi kendaraan dalam gambar.</li>
        </ul>
        <p style='color:#80deea;'>Pilih mode analisis di sidebar kiri untuk memulai.</p>
    </div>
    """, unsafe_allow_html=True)

# ================================
# Halaman KLASIFIKASI HEWAN
# ================================
elif menu == "Klasifikasi Hewan" and classifier is not None:
    uploaded_file = st.file_uploader("Unggah Gambar 📤", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="🖼️ Gambar yang diunggah", use_container_width=True)
        with st.spinner("🤖 Menganalisis data visual..."):
            try:
                input_shape = classifier.input_shape[1:3]
                img_resized = img.resize(input_shape)
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                prediction = classifier.predict(img_array)
                class_index = int(prediction[0][0] > 0.5)
                kelas = label_mapping.get(class_index, "Unknown")
                confidence = float(np.max(prediction))

                st.markdown(f"""
                <div class="result-box">
                    <h3>🧩 Kelas Teridentifikasi: {kelas}</h3>
                    <p>📊 Confidence: {confidence*100:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Gagal melakukan klasifikasi: {e}")
    else:
        st.info("Unggah gambar untuk memulai klasifikasi futuristik 🚀")

# ================================
# Halaman DETEKSI MOBIL (YOLO)
# ================================
elif menu == "Deteksi Mobil (YOLO)" and YOLO_AVAILABLE and yolo_model is not None:
    uploaded_file = st.file_uploader("Unggah Gambar 📤", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="🖼️ Gambar yang diunggah", use_container_width=True)
        with st.spinner("🚗 Memindai objek dengan YOLO..."):
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
                        <div class="result-box">
                            <h3>✅ {obj} terdeteksi!</h3>
                            <p>📊 Confidence: {conf*100:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("🚫 Tidak ada kendaraan terdeteksi.")
                st.image(result_img, caption="🔍 Hasil Deteksi", use_container_width=True)
            except Exception as e:
                st.error(f"Gagal mendeteksi objek: {e}")
    else:
        st.info("Unggah gambar untuk memulai deteksi kendaraan futuristik 🚘")
