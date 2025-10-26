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
# Menu
# ================================
menu = st.sidebar.selectbox(
    "Pilih Halaman 🖥️:",
    ["🏠 Home", "🐾 Klasifikasi Hewan", "🚗 Deteksi Kendaraan (YOLO)"]
)

# ================================
# Tema Dinamis
# ================================
if menu == "🐾 Klasifikasi Hewan":
    background = """
        background: linear-gradient(120deg, #c4f5d2, #aee8ff);
        color: #002b36;
    """
    accent_color = "#2ecc71"
    glow_color = "#00cc99"
    font_family = "'Poppins', sans-serif"

elif menu == "🚗 Deteksi Kendaraan (YOLO)":
    background = """
        background: linear-gradient(135deg, #00111a, #002b4f, #004d99);
        color: #e0f7fa;
    """
    accent_color = "#00ffff"
    glow_color = "#33ccff"
    font_family = "'Orbitron', sans-serif"

else:
    background = """
        background: linear-gradient(135deg, #e0f7fa, #ffffff);
        color: #001d3d;
    """
    accent_color = "#0077b6"
    glow_color = "#66d9ff"
    font_family = "'Poppins', sans-serif"

# ================================
# Styling Futuristik
# ================================
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500&family=Poppins:wght@400;600&display=swap');

    body {{
        {background}
        background-attachment: fixed;
        background-size: cover;
    }}
    .stApp {{
        background: rgba(255,255,255,0.05);
        border-radius: 20px;
        padding: 25px;
        backdrop-filter: blur(10px);
        box-shadow: 0 0 25px {glow_color};
    }}
    h1 {{
        text-align: center;
        font-size: 55px;
        font-family: {font_family};
        color: {accent_color};
        text-shadow: 0 0 30px {glow_color}, 0 0 50px {accent_color};
        letter-spacing: 2px;
        animation: glow 3s ease-in-out infinite alternate;
    }}
    @keyframes glow {{
        from {{ text-shadow: 0 0 10px {glow_color}, 0 0 20px {accent_color}; }}
        to {{ text-shadow: 0 0 25px {accent_color}, 0 0 45px {glow_color}; }}
    }}
    h3, h4, p {{
        font-family: {font_family};
        color: inherit;
        text-align: center;
    }}
    .kotak-hewan {{
        background: rgba(173, 216, 230, 0.25);
        border-left: 6px solid {accent_color};
        padding: 15px;
        border-radius: 12px;
        margin-top: 10px;
        box-shadow: 0 0 15px {accent_color};
    }}
    .kotak-mobil {{
        background: rgba(0, 255, 255, 0.15);
        border-left: 6px solid {accent_color};
        padding: 15px;
        border-radius: 12px;
        margin-top: 10px;
        box-shadow: 0 0 15px {accent_color};
    }}
    .particle {{
        position: fixed;
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: {glow_color};
        animation: float 6s ease-in-out infinite;
        opacity: 0.4;
    }}
    @keyframes float {{
        0% {{ transform: translateY(0) scale(1); opacity: 0.3; }}
        50% {{ transform: translateY(-25px) scale(1.4); opacity: 0.7; }}
        100% {{ transform: translateY(0) scale(1); opacity: 0.3; }}
    }}
    </style>
""", unsafe_allow_html=True)

# ================================
# HEADER
# ================================
st.markdown(f"""
<h1>🤖 APLIKASI DETEKSI & KLASIFIKASI GAMBAR</h1>
<h3>Jurusan Statistika — Universitas Syiah Kuala</h3>
<h4>NPM: <b>2208108010026</b> | Oleh: <b>Izzul Akrami</b></h4>
<hr style='border: 1px solid {accent_color};'>
""", unsafe_allow_html=True)

# ================================
# HALAMAN HOME
# ================================
if menu == "🏠 Home":
    st.markdown(f"""
    <div style='text-align:center;'>
        <h2>Selamat Datang di Dashboard AI Futuristik 🚀</h2>
        <p>Aplikasi ini menggabungkan dua dunia berbeda:</p>
        <ul style='text-align:left; display:inline-block; text-align:justify;'>
            <li>🐾 <b>Klasifikasi Hewan</b> — mengenali <i>Kucing</i> dan <i>Anjing</i> dengan nuansa alam digital.</li>
            <li>🚗 <b>Deteksi Kendaraan</b> — mendeteksi mobil & truk dalam gaya cybernetic.</li>
        </ul>
        <p>Pilih halaman di sidebar untuk mulai eksplorasi AI!</p>
    </div>
    """, unsafe_allow_html=True)

# ================================
# HALAMAN KLASIFIKASI HEWAN
# ================================
elif menu == "🐾 Klasifikasi Hewan" and classifier is not None:
    uploaded_file = st.file_uploader("Unggah Gambar Hewan 📤", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="📸 Gambar yang Diupload", use_container_width=True)
        with st.spinner("🔍 AI sedang menganalisis..."):
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
                lokasi = lokasi_mapping.get(kelas, "Area Tidak Dikenal")
                st.markdown(f"""
                <div class='kotak-hewan'>
                    <h3>✅ {kelas}</h3>
                    <p>📍 Lokasi: {lokasi}</p>
                    <p>📊 Confidence: {confidence*100:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Gagal melakukan klasifikasi: {e}")
    else:
        st.info("Unggah gambar hewan untuk memulai klasifikasi. 🐾")

# ================================
# HALAMAN DETEKSI KENDARAAN (YOLO)
# ================================
elif menu == "🚗 Deteksi Kendaraan (YOLO)" and YOLO_AVAILABLE and yolo_model is not None:
    uploaded_file = st.file_uploader("Unggah Gambar Kendaraan 📤", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="📸 Gambar yang Diupload", use_container_width=True)
        with st.spinner("⚙️ Sistem robotik sedang mendeteksi objek..."):
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
                            <h3>✅ {obj} Terdeteksi!</h3>
                            <p>📊 Confidence: {conf*100:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("🚫 Tidak ada mobil atau truk terdeteksi.")
                st.image(result_img, caption="🧾 Hasil Deteksi", use_container_width=True)
            except Exception as e:
                st.error(f"Gagal mendeteksi objek: {e}")
    else:
        st.info("Unggah gambar kendaraan untuk mulai deteksi. 🚗")
