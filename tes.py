# from flask import Flask, request, send_from_directory
# import os
# import time

# app = Flask(__name__)

# # Direktori untuk menyimpan gambar yang di-upload
# UPLOAD_FOLDER = './images/'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Fungsi untuk memberikan nama custom pada gambar dengan ekstensi .jpg
# def get_custom_filename():
#     return "imageFile.jpg"

# # Membaca gambar yang di-upload dan menyimpannya
# @app.route('/upload', methods=['POST'])
# def upload_image():
#     # Memeriksa apakah file gambar ada dalam request
#     if 'image' not in request.files:
#         return 'No image part'

#     imageFile = request.files['image']
#     if imageFile.filename == '':
#         return 'No selected file'

#     # Mendapatkan nama custom untuk file
#     filename = get_custom_filename()
#     image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

#     # Hapus file sebelumnya jika ada
#     if os.path.exists(image_path):
#         os.remove(image_path)

#     imageFile.save(image_path)
#     return f"Image uploaded successfully. Path: {image_path}"

# # Untuk menampilkan gambar dari server
# @app.route('/images/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8004, debug=True)

from flask import Flask, request, send_from_directory
import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
import joblib  # Untuk memuat model jika disimpan dengan joblib

app = Flask(__name__)

# Direktori untuk menyimpan gambar yang di-upload
UPLOAD_FOLDER = './images/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Muat model dan face cascade
model = joblib.load('D:\\Belajar Machine Learning\\famscreen_ml\\FamScreen_ML\\famscreen_model.pkl')  
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Fungsi untuk memberikan nama custom pada gambar dengan ekstensi .jpg
def get_custom_filename():
    return "imageFile.jpg"

# Fungsi untuk mengonversi umur menjadi kategori umur
def age_category_to_label(age_category):
    if age_category == 0:
        return "Anak-anak"
    elif age_category == 1:
        return "Remaja"
    elif age_category == 2:
        return "Dewasa"
    return "Tidak Valid"


# Fungsi untuk ekstraksi fitur LBP
def extract_lbp_features(image, radius=1, n_points=8) :
    # Hitung LBP pada gambar
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    
    # Hitung histogram dari LBP
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    
    # Normalisasi histogram
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)  # Normalisasi agar jumlah histogram = 1
    return lbp_hist


# Fungsi untuk augmentasi gambar
def augment_image(image):
       # Flip horizontal secara acak
       if random.choice([True, False]):
              image = cv2.flip(image, 1)
       # Rotasi 10 derajat secara acak
       if random.choice([True, False]):
              h, w = image.shape
              center = (w // 2, h // 2)
              M = cv2.getRotationMatrix2D(center, angle=random.uniform(-10, 10), scale=1.0)
              image = cv2.warpAffine(image, M, (w, h))
       return image


# Fungsi untuk mendeteksi wajah dan prediksi umur
def detect_and_predict(image_path, model, face_cascade, radius=1, n_points=8):
    # Membaca gambar dan konversi ke grayscale
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah dalam gambar
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Jika wajah terdeteksi
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Crop wajah
            face_image = gray_image[y:y+h, x:x+w]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_image_resized = cv2.resize(face_image, (150, 150))

            # Terapkan augmentasi pada wajah
            augmented_face_image = augment_image(face_image_resized)

            # Ekstraksi fitur LBP dari wajah yang sudah diaugmentasi
            lbp_features = extract_lbp_features(augmented_face_image, radius=radius, n_points=n_points)

            # Prediksi menggunakan model Random Forest
            prediction = model.predict([lbp_features])  # Model menerima array 2D, jadi fitur LBP harus dalam list
            predicted_age_category = prediction[0]
            
            # Mengonversi angka prediksi menjadi kategori usia (label teks)
            predicted_age_label = age_category_to_label(predicted_age_category)

            # Visualisasi hasil
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(f'Prediksi Umur: {predicted_age_label}')
            plt.axis('off')
            plt.show()

            return predicted_age_category, predicted_age_label
    else:
        print("Tidak ada wajah yang terdeteksi dalam gambar.")
        return None, None

# Membaca gambar yang di-upload dan menyimpannya
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return 'No image part'

    imageFile = request.files['image']
    if imageFile.filename == '':
        return 'No selected file'

    filename = get_custom_filename()
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(image_path):
        os.remove(image_path)

    imageFile.save(image_path)
    return f"Image uploaded successfully. Path: {image_path}"

# Endpoint prediksi umur
@app.route('/prediksi')
def toPredict():
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], "imageFile.jpg")
    if not os.path.exists(image_path):
        return "Image not found. Please upload an image first."

    detected_age_category, predicted_age_label = detect_and_predict(image_path, model, face_cascade)
    if detected_age_category is not None:
        return f"{predicted_age_label}"
    else:
        return "Tidak ada wajah yang terdeteksi dalam gambar."

# Untuk menampilkan gambar dari server
@app.route('/images/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8004, debug=True)
