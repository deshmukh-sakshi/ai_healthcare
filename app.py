from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from model_utils import download_model
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variable for model
MODEL = None

def load_model_global():
    """Initialize the global model."""
    global MODEL
    if MODEL is None:
        try:
            model_path = download_model()
            MODEL = load_model(model_path)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_tumor(image_path):
    """
    Make prediction using the loaded model
    """
    global MODEL
    if MODEL is None:
        load_model_global()
    
    # Load and preprocess the image
    img = load_img(image_path, target_size=(256, 256), color_mode='grayscale')
    img_array = img_to_array(img)
    
    # Normalize and prepare input
    img_normalized = img_array / 255.0
    img_normalized = np.expand_dims(img_normalized, axis=0)
    
    # Make prediction
    predictions = MODEL.predict(img_normalized)
    classes = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']
    
    # Get prediction results
    predicted_class = classes[np.argmax(predictions[0])]
    confidence_scores = {
        class_name: float(prob) * 100 
        for class_name, prob in zip(classes, predictions[0])
    }
    
    return predicted_class, confidence_scores

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Make prediction
            predicted_class, confidence_scores = predict_tumor(filepath)
            
            return render_template('result.html',
                                prediction=predicted_class,
                                confidence_scores=confidence_scores,
                                image_path=f'uploads/{filename}')
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

# Initialize model when starting the app
@app.before_first_request
def initialize():
    load_model_global()

if __name__ == '__main__':
    app.run(debug=True) 