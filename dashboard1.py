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
        background: linear-gradient(135deg, #e0f7fa, #b2ebf2, #ffffff);
        color: #0a192f;
    """
    accent_color = "#00bcd4"
    font_family = "'Comfortaa', sans-serif"
    title_color = "#00838f"
    subtitle_color = "#004d40"

elif menu == "🚗 Deteksi Kendaraan (YOLO)":
    background = """
        background: linear-gradient(135deg, #e3f2fd, #bbdefb, #90caf9);
        color: #001d3d;
    """
    accent_color = "#1565c0"
    font_family = "'Orbitron', sans-serif"
    title_color = "#0d47a1"
    subtitle_color = "#003c8f"

else:
    background = """
        background: linear-gradient(135deg, #ffffff, #e0f7fa, #b3e5fc);
        color: #001d3d;
    """
    accent_color = "#0077b6"
    font_family = "'Poppins', sans-serif"
    title_color = "#01579b"
    subtitle_color = "#0277bd"

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
        background: rgba(255,255,255,0.15);
        border-radius: 15px;
        padding: 20px;
        backdrop-filter: blur(10px);
    }}
    .stButton>button {{
        background-color: {accent_color};
        color: white;
        border-radius: 10px;
        border: none;
        font-weight: bold;
    }}
    .stSidebar {{
        background: rgba(0,0,0,0.05);
    }}
    h1, h2, h3, h4, p {{
        font-family: {font_family};
    }}
    .kotak-hewan {{
        background: rgba(173, 216, 230, 0.3);
        border-left: 5px solid {accent_color};
        padding: 15px;
        border-radius: 12px;
        margin-top: 10px;
        box-shadow: 0 0 15px {accent_color};
    }}
    .kotak-mobil {{
        background: rgba(173, 216, 230, 0.25);
        border-left: 5px solid {accent_color};
        padding: 15px;
        border-radius: 12px;
        margin-top: 10px;
        box-shadow: 0 0 15px {accent_color};
    }}
    </style>
""", unsafe_allow_html=True)

# ================================
# HEADER (Judul berbeda tiap halaman)
# ================================
if menu == "🏠 Home":
    title_text = "🤖 APLIKASI DETEKSI & KLASIFIKASI GAMBAR"
elif menu == "🐾 Klasifikasi Hewan":
    title_text = "🐾 MODE KLASIFIKASI HEWAN"
elif menu == "🚗 Deteksi Kendaraan (YOLO)":
    title_text = "🚗 MODE DETEKSI KENDARAAN (YOLO)"

st.markdown(f"""
<h1 style='text-align:center; color:{title_color}; font-size: 50px; text-shadow: 0 0 18px {accent_color};'>
{title_text}
</h1>
<h3 style='text-align:center; color:{subtitle_color};'>Jurusan Statistika — Universitas Syiah Kuala</h3>
<h4 style='text-align:center; color:{subtitle_color};'>NPM: <b>2208108010026</b> | Oleh: <b>Izzul Akrami</b></h4>
<hr style='border: 1px solid {accent_color};'>
""", unsafe_allow_html=True)

# ================================
# HALAMAN HOME
# ================================
if menu == "🏠 Home":
    st.markdown(f"""
    <div style='text-align:center;'>
        <h2>Selamat Datang di Dashboard AI 🚀</h2>
        <p>Aplikasi ini menggunakan <b>Deep Learning</b> dan <b>YOLO</b> untuk mengenali dua dunia berbeda:</p>
        <ul style='text-align:left; display:inline-block; text-align:justify;'>
            <li>🐾 <b>Klasifikasi Hewan</b> — membedakan antara <i>Kucing</i> dan <i>Anjing</i>.</li>
            <li>🚗 <b>Deteksi Kendaraan</b> — mendeteksi mobil dan truk secara otomatis.</li>
        </ul>
        <p style='font-size:18px; color:{accent_color}; margin-top:20px;'>
            Pilih mode di sidebar untuk memulai!
        </p>
        <h1 style='color:{accent_color}; font-size:55px; margin-top:15px; text-shadow: 0 0 20px {accent_color}; font-family:{font_family};'>
            😊 SELAMAT MENGGUNAKAN 😊
        </h1>
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
                    <h3 style='color:{accent_color};'>✅ {kelas}</h3>
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
                            <h3 style='color:{accent_color};'>✅ {obj} Terdeteksi!</h3>
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
