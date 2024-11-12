from flask import Flask, request, send_from_directory, jsonify
import os
import time
import cv2
import numpy as np
import joblib
from matplotlib import pyplot as plt
from skimage.feature import local_binary_pattern

app = Flask(__name__)

# Direktori untuk menyimpan gambar yang di-upload
UPLOAD_FOLDER = './images/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Fungsi untuk memberikan nama custom pada gambar
def get_custom_filename(filename):
    # Mengambil ekstensi dari nama file
    ext = filename.split('.')[-1]
    # Membuat nama custom dengan timestamp dan ekstensi file
    custom_name = f"{int(time.time())}.{ext}"  
    return custom_name

# Fungsi untuk mengubah angka prediksi menjadi kategori usia (label teks)
def age_category_to_label(age_category):
    if age_category == 0:
        return "Anak-anak"
    elif age_category == 1:
        return "Remaja"
    elif age_category == 2:
        return "Dewasa"
    elif age_category == 3:
        return "Lansia"
    return "Tidak Valid"

# Fungsi untuk ekstraksi fitur LBP
def extract_lbp_features(image, radius=1, n_points=8):
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    n_bins = int(lbp.max() + 1)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    lbp_hist = lbp_hist.astype('float')
    lbp_hist /= (lbp_hist.sum() + 1e-6)  # Normalisasi histogram
    return lbp_hist.reshape(1, -1)  # Mengubah bentuk histogram menjadi array 2D

# Fungsi untuk mendeteksi wajah dan memprediksi kategori usia
def detect_and_predict_from_image(image_path, model, face_cascade):
    # Membaca gambar dan mengkonversinya ke RGB
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Deteksi wajah dalam gambar
    faces = face_cascade.detectMultiScale(image_rgb, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Jika wajah terdeteksi
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Crop wajah dan ubah ukurannya
            face_image = image_rgb[y:y+h, x:x+w]
            face_image_resized = cv2.resize(face_image, (100, 100))

            # Ekstraksi fitur LBP
            lbp_features = extract_lbp_features(face_image_resized)

            # Prediksi kategori usia dengan model
            prediction = model.predict(lbp_features)
            predicted_age_category = prediction[0]

            # Mengonversi angka prediksi menjadi kategori usia
            predicted_age_label = age_category_to_label(predicted_age_category)

            return predicted_age_label
    else:
        return "Tidak ada wajah yang terdeteksi"

# Membaca model yang sudah disimpan
model = joblib.load('D:\\Belajar Machine Learning\\ML Famscreen\\FamScreen_ML\\famscreen_model.pkl')

# Inisialisasi face cascade untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Membaca gambar yang di-upload dan menyimpannya
@app.route('/upload', methods=['POST'])
def upload_image():
    # Memeriksa apakah file gambar ada dalam request
    if 'image' not in request.files:
        return 'No image part', 400

    imageFile = request.files['image']
    if imageFile.filename == '':
        return 'No selected file', 400

    # Mendapatkan nama custom untuk file
    filename = get_custom_filename(imageFile.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    imageFile.save(image_path)

    # Proses gambar dan deteksi usia
    predicted_age_label = detect_and_predict_from_image(image_path, model, face_cascade)
    
    return jsonify({
        "message": "Image uploaded and processed successfully",
        "predicted_age_label": predicted_age_label,
        "image_path": image_path
    })

# Untuk menampilkan gambar dari server
@app.route('/images/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Membuat direktori upload jika belum ada
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    app.run(port=8004, debug=True)
