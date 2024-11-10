from flask import Flask, request, send_from_directory
import os
import time

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

# Membaca gambar yang di-upload dan menyimpannya
@app.route('/upload', methods=['POST'])
def upload_image():
    # Memeriksa apakah file gambar ada dalam request
    if 'image' not in request.files:
        return 'No image part'

    imageFile = request.files['image']
    if imageFile.filename == '':
        return 'No selected file'

    # Mendapatkan nama custom untuk file
    filename = get_custom_filename(imageFile.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    imageFile.save(image_path)
    return f"Image uploaded successfully. Path: {image_path}"

# Untuk menampilkan gambar dari server
@app.route('/images/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(port=8004, debug=True)
