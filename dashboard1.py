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
# Styling Dashboard: Tema Biru Malam Elegan
# ================================
st.set_page_config(
    page_title="📷 Aplikasi Deteksi & Klasifikasi",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    body {
        background-image: url('https://www.toptal.com/designers/subtlepatterns/uploads/geometry2.png');
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        background-color: #0a0f24;
        color: white;
    }
    .stApp {
        background-color: rgba(10, 15, 36, 0.90);
        padding: 25px;
        border-radius: 20px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #007bff, #00c6ff);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: bold;
        padding: 10px 18px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #00c6ff, #007bff);
        transform: scale(1.03);
    }
    .stSidebar {
        background-color: rgba(15, 25, 55, 0.95);
        color: white;
    }
    h1, h2, h3, p, li {
        color: white !important;
    }
    .kotak-hewan {
        background: rgba(0, 123, 255, 0.2);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border: 1px solid rgba(0, 123, 255, 0.5);
    }
    .kotak-mobil {
        background: rgba(0, 255, 150, 0.15);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border: 1px solid rgba(0, 255, 150, 0.4);
    }
    /* Logo USK pojok kiri atas */
    .logo-usk {
        position: fixed;
        top: 15px;
        left: 20px;
        width: 90px;
        z-index: 9999;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Tambahkan logo USK (gunakan logo dari situs resmi)
st.markdown("""
    <img src="https://upload.wikimedia.org/wikipedia/id/7/76/Lambang_Universitas_Syiah_Kuala.png" class="logo-usk">
""", unsafe_allow_html=True)

# ================================
# UI
# ================================
st.markdown("<h1 style='text-align:center; color:#00c6ff;'>📷 Aplikasi Deteksi & Klasifikasi</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#cce7ff;'>Dikembangkan oleh: <b>Izzul Akrami</b> | Universitas Syiah Kuala</p>", unsafe_allow_html=True)
st.divider()

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
        <h2 style='color:#00c6ff;'>Selamat Datang di Aplikasi Deteksi & Klasifikasi Gambar</h2>
        <p style='color:#d0e7ff;'>Aplikasi ini dapat mengenali dua jenis objek:</p>
        <ul style='text-align:left; display:inline-block; text-align:justify; color:#e8f1ff;'>
            <li>🐱 <b>Klasifikasi Hewan:</b> Membedakan antara <i>Kucing</i> dan <i>Anjing</i>.</li>
            <li>🚗 <b>Deteksi Mobil (YOLO):</b> Mendeteksi keberadaan mobil di dalam gambar.</li>
        </ul>
        <p style='color:#b3d4ff;'>Gunakan menu di sebelah kiri untuk memilih mode yang diinginkan.<br>
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
