from flask import Flask, request, jsonify, send_from_directory, url_for
import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
import joblib
from celery import Celery
import logging

# Setup Flask dan Celery
app = Flask(__name__)
UPLOAD_FOLDER = './images/'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Maksimum ukuran file 16 MB
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'  # URL broker Redis
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'  # Hasil juga disimpan di Redis

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# Logging setup
logging.basicConfig(level=logging.INFO)

# Pastikan folder upload ada
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load model dan face detector
model = joblib.load('famscreen_model.pkl')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Fungsi utilitas
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def enhance_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def extract_lbp_features(image, radius=2, n_points=16):
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)
    return lbp_hist

def age_category_to_label(age_category):
    if age_category == 0:
        return "Anak-anak"
    elif age_category == 1:
        return "Remaja"
    elif age_category == 2:
        return "Dewasa"
    return "Tidak Valid"

# Tugas Celery untuk pemrosesan gambar
@celery.task
def process_image_task(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced_img = enhance_image(gray_image)
    faces = face_cascade.detectMultiScale(enhanced_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            face_image = enhanced_img[y:y+h, x:x+w]
            face_image_resized = cv2.resize(face_image, (256, 256))
            lbp_features = extract_lbp_features(face_image_resized)
            prediction = model.predict([lbp_features])
            predicted_age_label = age_category_to_label(prediction[0])
            return {'status': 'success', 'prediction': predicted_age_label}

    return {'status': 'failure', 'message': 'No face detected in the image.'}

# Endpoint untuk upload gambar dan memulai tugas pemrosesan
@app.route('/upload', methods=['POST'])
def upload_and_process():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file found in request'}), 400

    image_file = request.files['image']
    if image_file.filename == '' or not allowed_file(image_file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    # Simpan file yang diunggah
    filename = "uploaded_image.jpg"
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(image_path):
        os.remove(image_path)
    image_file.save(image_path)

    # Mulai tugas Celery
    task = process_image_task.apply_async(args=[image_path])
    return jsonify({'status': 'processing', 'task_id': task.id}), 202

# Endpoint untuk memeriksa status tugas
@app.route('/status/<task_id>', methods=['GET'])
def get_status(task_id):
    task = process_image_task.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {'state': task.state, 'status': 'Task is pending...'}
    elif task.state == 'SUCCESS':
        response = {'state': task.state, 'result': task.result}
    else:
        response = {'state': task.state, 'status': str(task.info)}
    return jsonify(response)

# Endpoint untuk health check
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'face_cascade_loaded': not face_cascade.empty()
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8004, debug=False)
