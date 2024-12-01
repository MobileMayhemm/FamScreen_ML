from flask import Flask, request, jsonify, send_from_directory
import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import local_binary_pattern
import joblib

app = Flask(__name__)
UPLOAD_FOLDER = './images/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model dan face detector
model = joblib.load('New/famscreen_model.pkl')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def enhance_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def extract_lbp_features(image, radius=2, n_points=24):
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)
    return lbp_hist

def predict_age_category(image_path, model):
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Gambar tidak dapat dibaca atau kosong: {image_path}")
    
    # Preprocessing
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced_img = enhance_image(gray_image)
    
    # Detect face
    faces = face_cascade.detectMultiScale(enhanced_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        cropped_face = enhanced_img[y:y+h, x:x+w]
        
        # Resize and extract features
        face_image_resized = cv2.resize(cropped_face, (256, 256))
        lbp_features = extract_lbp_features(face_image_resized)
        lbp_features = np.array(lbp_features).reshape(1, -1)
        
        # Predict
        prediction = model.predict(lbp_features)
        return prediction[0]  # Return predicted label
    
    return None  # No face detected

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
                'prediction': int(prediction),  # Convert NumPy value to standard Python int
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