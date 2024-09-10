from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from PIL import Image
import os
import tensorflow as tf
import numpy as np
from keras.models import load_model  # type: ignore

app = Flask(__name__, static_folder='static', template_folder='static')
CORS(app)  # Enable CORS for all domains

UPLOAD_FOLDER = './uploaded_images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Image dimensions
IMG_HEIGHT = 256
IMG_WIDTH = 256

def normalize_test_image(input_image):
    input_image = (input_image / 127.5) - 1
    return input_image

def resize_test_image(input_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image

def load_test_image(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    input_image = tf.cast(image, tf.float32)
    return input_image

def load_image_test(image_file):
    input_image = load_test_image(image_file)
    input_image = resize_test_image(input_image, IMG_HEIGHT, IMG_WIDTH)
    input_image = normalize_test_image(input_image)
    return input_image

def save_image(image_tensor, output_path):
    # Rescale the image from [-1, 1] to [0, 1]
    image = (image_tensor[0].numpy() * 0.5 + 0.5) * 255
    image = image.astype(np.uint8)
    # Save as JPEG
    img = Image.fromarray(image)
    img.save(output_path)

def process_and_save_image(model, input_image_path):
    # Load and process the input image
    input_image = load_image_test(input_image_path)
    input_image = np.expand_dims(input_image, axis=0)

    # Get the prediction
    prediction = model(input_image, training=True)

    # Get the output file path
    output_image_path = os.path.join(UPLOAD_FOLDER, 'canvas_image_output.jpg')

    # Save the predicted image
    save_image(prediction, output_image_path)

    print(f"Output saved to {output_image_path}")

# Load the model
model = load_model("./model_000145.h5")

@app.route('/')
def index():
    return render_template('index.html')  # Serve the index.html file

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        image = Image.open(image_file.stream)
        
        # Convert to RGB (to handle any possible transparency in PNGs)
        image = image.convert('RGB')
        
        # Create the file path and save the image as JPEG
        file_name = os.path.splitext(image_file.filename)[0] + '.jpg'  # Replace extension with .jpg
        file_path = os.path.join(UPLOAD_FOLDER, file_name)
        image.save(file_path, 'JPEG')

        input_image_path = os.path.join(UPLOAD_FOLDER, file_name)
        process_and_save_image(model, input_image_path)
        
        image_path = os.path.join(UPLOAD_FOLDER, 'canvas_image_output.jpg')
        if os.path.exists(image_path):
            return send_file(image_path, mimetype='image/jpeg')
        else:
            return jsonify({'error': 'Image not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
