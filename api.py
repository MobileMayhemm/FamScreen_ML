from flask import Flask, request, jsonify, send_from_directory
import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import joblib

app = Flask(__name__)
UPLOAD_FOLDER = './images/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model dan face detector
model = joblib.load('New/famscreen_model.pkl')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Fungsi untuk melakukan enhancement pada citra
def enhance_image(image):
       """
       Fungsi untuk meningkatkan kualitas citra menggunakan teknik sharpening.

       Parameter:
       image : numpy array
              Citra input, bisa berupa citra berwarna (BGR) atau grayscale.

       Return:
       enhanced_img : numpy array
              Citra hasil setelah dilakukan enhancement (sharpening).
       """

       # Periksa apakah citra memiliki 3 kanal (citra berwarna)
       if len(image.shape) == 3:  
              # Jika berwarna (BGR), ubah menjadi grayscale
              gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       else:
              # Jika sudah grayscale, gunakan langsung
              gray = image  

       # Definisikan kernel sharpening (ukuran 3x3)
       # Nilai positif di tengah meningkatkan kontras, nilai negatif di sekitar meredam noise
       kernel = np.array([
              [0, -0.5, 0],
              [-0.5, 3, -0.5],
              [0, -0.5, 0]
       ])

       # Aplikasikan filter kernel pada citra grayscale untuk melakukan sharpening
       enhanced_img = cv2.filter2D(gray, -1, kernel)

       # Kembalikan citra yang telah ditingkatkan
       return enhanced_img

# Inisialisasi Mediapipe Face Mesh untuk deteksi wajah
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
       static_image_mode=True,               # Mode gambar statis (tidak video streaming)
       max_num_faces=1,                      # Maksimal mendeteksi 1 wajah
       refine_landmarks=True,                # Menggunakan landmark yang lebih presisi
       min_detection_confidence=0.5          # Kepercayaan minimum deteksi wajah
)

def detect_face_mediapipe(img):
       """
       Fungsi untuk mendeteksi wajah pada citra menggunakan Mediapipe Face Mesh.

       Parameter:
       img : numpy array
              Citra input dengan format BGR.

       Return:
       cropped_face : numpy array
              Citra hasil crop berdasarkan bounding box wajah.
              Return None jika tidak ada wajah terdeteksi.
       """
       
       # Konversi citra dari BGR ke RGB, karena Mediapipe hanya mendukung format RGB
       rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

       # Proses deteksi wajah menggunakan Mediapipe
       results = face_mesh.process(rgb_img)

       # Jika tidak ada wajah yang terdeteksi, kembalikan None
       if not results.multi_face_landmarks:
              return None

       # Ambil landmark dari wajah pertama yang terdeteksi
       face_landmarks = results.multi_face_landmarks[0]

       # Hitung koordinat landmark dalam piksel (dikonversi dari relatif ke ukuran asli gambar)
       h, w = img.shape[:2]  # Ambil dimensi tinggi (h) dan lebar (w) dari gambar
       landmarks_points = [
              (int(landmark.x * w), int(landmark.y * h))  # Ubah koordinat relatif ke piksel
              for landmark in face_landmarks.landmark
       ]

       # Hitung bounding box wajah berdasarkan koordinat landmark
       x_min = min([point[0] for point in landmarks_points])  # Koordinat X minimum
       y_min = min([point[1] for point in landmarks_points])  # Koordinat Y minimum
       x_max = max([point[0] for point in landmarks_points])  # Koordinat X maksimum
       y_max = max([point[1] for point in landmarks_points])  # Koordinat Y maksimum

       # Potong (crop) gambar berdasarkan bounding box wajah
       cropped_face = img[y_min:y_max, x_min:x_max]

       return cropped_face

# Fungsi untuk ekstraksi fitur LBP (Local Binary Pattern)
def extract_lbp_features(image, radius=3, n_points=24):
       """
       Fungsi untuk mengekstraksi fitur LBP dari citra dan menghasilkan histogram yang ter-normalisasi.

       Parameter:
       image : numpy array
              Citra input dalam format grayscale.
       radius : int
              Radius untuk menghitung LBP (default = 3).
       n_points : int
              Jumlah titik yang digunakan untuk LBP (default = 24).

       Return:
       lbp_hist : numpy array
              Histogram ter-normalisasi dari nilai LBP.
       """

       # Validasi input: Periksa apakah citra valid
       if image is None or image.size == 0:
              raise ValueError("Gambar yang diteruskan kosong atau tidak valid")

       # Hitung LBP pada citra
       # Parameter `method='uniform'` digunakan untuk menghasilkan pola LBP yang lebih kompak
       lbp = local_binary_pattern(image, n_points, radius, method="uniform")

       # Hitung histogram dari nilai-nilai LBP
       # Bins mencakup semua pola unik yang mungkin, ditambah dua pola tambahan untuk metode "uniform"
       lbp_hist, _ = np.histogram(
              lbp.ravel(),                # Flatten citra LBP menjadi satu dimensi
              bins=np.arange(0, n_points + 3),  # Bin dari 0 hingga (n_points + 2)
              range=(0, n_points + 2)    # Rentang nilai LBP
       )

       # Normalisasi histogram agar total probabilitas = 1
       # Normalisasi penting untuk membuat fitur tidak bergantung pada ukuran citra
       lbp_hist = lbp_hist.astype("float")
       lbp_hist /= (lbp_hist.sum() + 1e-6)  # Hindari pembagian dengan nol

       return lbp_hist

# Fungsi untuk mendeteksi wajah menggunakan MediaPipe


# Fungsi untuk memproses gambar dan memprediksi kategori usia
def predict_age_category(image_path, model):
    # Membaca gambar dari path
    image = cv2.imread(image_path)
    
    if image is None:  # Cek apakah gambar valid
        raise ValueError(f"Gambar tidak dapat dibaca atau kosong: {image_path}")
    
    # Mengubah gambar menjadi grayscale
    enhanced_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Deteksi wajah dan potong bagian wajah
    cropped_face = detect_face_mediapipe(enhanced_img)
    
    if cropped_face is None:  # Jika tidak ada wajah yang terdeteksi
        print(f"Tidak ada wajah yang terdeteksi pada gambar: {image_path}")
        return None, image, None
    
    if cropped_face.size == 0:  # Cek apakah wajah yang terpotong kosong
        print(f"Wajah yang terpotong kosong: {image_path}")
        return None, image, None
    
    # Ekstraksi fitur LBP dari wajah yang terdeteksi
    try:
        lbp_features = extract_lbp_features(cropped_face)
    except ValueError as e:  # Menangani error jika ekstraksi gagal
        print(f"Error saat ekstraksi LBP pada gambar {image_path}: {e}")
        return None, image, cropped_face
    
    # Mengubah fitur menjadi format 2D untuk prediksi
    lbp_features = np.array(lbp_features).reshape(1, -1)
    feature_names = df.columns[1:]  # Mengambil kolom fitur kecuali 'age_category'
    
    # Membuat DataFrame untuk prediksi
    lbp_features_df = pd.DataFrame(lbp_features, columns=feature_names)
    
    # Prediksi kategori usia menggunakan model
    age_category_pred = model.predict(lbp_features_df)
    
    return age_category_pred[0], image, cropped_face  # Mengembalikan kategori, gambar asli, dan hasil crop



@app.route('/upload', methods=['POST'])
def upload_and_predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file found in request'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save uploaded file
    filename = "uploaded_image.jpg"
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(image_path):
        os.remove(image_path)
    image_file.save(image_path)

    try:
        # Predict age category
        prediction = predict_age_category(image_path, model)
        
        if prediction is not None:
            return jsonify({
                'status': 'success',
                'prediction': int(prediction),  
            }), 200
        else:
            return jsonify({
                'status': 'failure',
                'message': 'No face detected in the image.',
                'image_url': f"http://{request.host}/images/{filename}"
            }), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8004, debug=True)
