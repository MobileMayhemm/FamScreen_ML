from flask import Flask, request, jsonify, send_from_directory
import os
import json
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import local_binary_pattern
import joblib

app = Flask(__name__)
UPLOAD_FOLDER = './images/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model dan face detector
model = joblib.load('famscreen_model.pkl')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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

    # Process and predict
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
            return jsonify({
                'status': 'success',
                'prediction': predicted_age_label,
            }), 200
    else:
        # Response jika tidak ada wajah terdeteksi
        return jsonify({
            'status': 'failure',
            'message': 'No face detected in the image.',
            'image_url': f"http://{request.host}/images/{filename}"
        }), 200   

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'face_cascade_loaded': not face_cascade.empty()
    }), 200

@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8004, debug=True)
