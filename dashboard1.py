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
        background: radial-gradient(circle at top left, #d4fc79, #96e6a1, #c9ffe5);
        color: #0b132b;
    """
    accent_color = "#22c1c3"
    font_family = "'Orbitron', sans-serif"
    title_text = "🐾 KLASIFIKASI HEWAN CERDAS (CAT VS DOG)"

elif menu == "🚗 Deteksi Kendaraan (YOLO)":
    background = """
        background: linear-gradient(135deg, #e3f2fd, #bbdefb, #e0f7fa);
        color: #002b36;
    """
    accent_color = "#00bcd4"
    font_family = "'Audiowide', sans-serif"
    title_text = "🚗 DETEKSI KENDARAAN FUTURISTIK (CAR & TRUCK)"

else:
    background = """
        background: linear-gradient(135deg, #c9e7ff, #e0f7fa, #ffffff);
        color: #001d3d;
    """
    accent_color = "#0077b6"
    font_family = "'Poppins', sans-serif"
    title_text = "🤖 APLIKASI DETEKSI & KLASIFIKASI GAMBAR"

# ================================
# Styling Dinamis
# ================================
st.markdown(f"""
    <style>
    body {{
        {background}
        background-attachment: fixed;
        background-size: cover;
    }}
    .stApp {{
        background: rgba(255,255,255,0.2);
        border-radius: 15px;
        padding: 20px;
        backdrop-filter: blur(15px);
    }}
    .stButton>button {{
        background-color: {accent_color};
        color: white;
        border-radius: 10px;
        border: none;
        font-weight: bold;
        transition: 0.3s;
    }}
    .stButton>button:hover {{
        background-color: white;
        color: {accent_color};
        border: 2px solid {accent_color};
    }}
    h1, h2, h3, h4, p {{
        font-family: {font_family};
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
        background: rgba(0, 188, 212, 0.15);
        border-left: 6px solid {accent_color};
        padding: 15px;
        border-radius: 12px;
        margin-top: 10px;
        box-shadow: 0 0 20px {accent_color};
    }}
    </style>
""", unsafe_allow_html=True)

# ================================
# HEADER
# ================================
st.markdown(f"""
<h1 style='text-align:center; color:{accent_color}; font-size: 55px; text-shadow: 0 0 20px {accent_color};'>
{title_text}
</h1>
<h3 style='text-align:center;'>Jurusan Statistika — Universitas Syiah Kuala</h3>
<h4 style='text-align:center;'>NPM: <b>2208108010026</b> | Oleh: <b>Izzul Akrami</b></h4>
<hr style='border: 1px solid {accent_color};'>
""", unsafe_allow_html=True)

# ================================
# HALAMAN HOME
# ================================
if menu == "🏠 Home":
    st.markdown(f"""
    <div style='text-align:center;'>
        <h2>Selamat Datang di Dashboard Futuristik 🚀</h2>
        <p>Aplikasi ini memanfaatkan <b>Deep Learning</b> dan <b>YOLO</b> untuk mengenali dua dunia berbeda:</p>
        <ul style='text-align:left; display:inline-block; text-align:justify;'>
            <li>🐾 <b>Klasifikasi Hewan</b> — membedakan antara <i>Kucing</i> dan <i>Anjing</i> dengan tema segar alami.</li>
            <li>🚗 <b>Deteksi Kendaraan</b> — mendeteksi mobil dan truk dengan nuansa terang futuristik dan robotik.</li>
        </ul>
        <p>Pilih mode di sidebar untuk memulai eksperimen AI!</p>
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
