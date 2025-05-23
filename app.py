from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import pytesseract
from PIL import Image
import base64
import io
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:3000",
            "https://your-app-frontend.onrender.com"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Configure Tesseract path
possible_paths = [
    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
    r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
    r'C:\Tesseract-OCR\tesseract.exe'
]

tesseract_path = None
for path in possible_paths:
    if os.path.exists(path):
        tesseract_path = path
        break

if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    logger.info("Tesseract found at: %s", tesseract_path)
else:
    logger.error("Tesseract not found. Please install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki")
    logger.error("After installation, make sure to add Tesseract to your system PATH")
    raise Exception("Tesseract OCR is not installed. Please install it first.")

def process_image(image_data):
    if not tesseract_path:
        raise Exception("Tesseract OCR is not installed. Please install it first.")

    try:
        # Convert base64 to image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert PIL Image to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to preprocess the image
        threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Perform text detection
        text = pytesseract.image_to_string(threshold)
        
        if not text.strip():
            return "No text detected in the image."
            
        return text.strip()
        
    except Exception as e:
        logger.error("Error processing image: %s", str(e))
        raise Exception(f"Error processing image: {str(e)}")

@app.route('/api/detect-text', methods=['POST'])
def detect_text():
    try:
        if not request.is_json:
            logger.error("Request is not JSON")
            return jsonify({'error': 'Request must be JSON'}), 400
            
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            logger.error("No image data provided")
            return jsonify({'error': 'No image data provided'}), 400
            
        logger.info("Processing image...")
        detected_text = process_image(image_data)
        logger.info("Text detection completed")
        return jsonify({'text': detected_text})
        
    except Exception as e:
        logger.error("Error in detect_text endpoint: %s", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask server...")
    app.run(debug=True, port=5000, host='0.0.0.0') 