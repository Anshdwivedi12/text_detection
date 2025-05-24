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
from dotenv import load_dotenv
from functools import wraps
import time
from werkzeug.utils import secure_filename

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_REQUESTS_PER_MINUTE = 60
ALLOWED_IMAGE_TYPES = {'image/jpeg', 'image/png', 'image/gif', 'image/bmp'}
MAX_IMAGE_DIMENSION = 4096  # Maximum width or height

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:3000",
            "https://text-detection-frontend.onrender.com",
            "https://text-detection-frontend.vercel.app"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Rate limiting decorator
def rate_limit(f):
    requests = {}
    
    @wraps(f)
    def decorated_function(*args, **kwargs):
        now = time.time()
        client_ip = request.remote_addr
        
        # Clean old requests
        requests[client_ip] = [req_time for req_time in requests.get(client_ip, [])
                             if now - req_time < 60]
        
        if len(requests.get(client_ip, [])) >= MAX_REQUESTS_PER_MINUTE:
            return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429
            
        requests.setdefault(client_ip, []).append(now)
        return f(*args, **kwargs)
    return decorated_function

# Configure Tesseract path
def get_tesseract_path():
    # First check environment variable
    if os.environ.get('TESSERACT_PATH'):
        path = os.environ.get('TESSERACT_PATH')
        if os.path.exists(path):
            return path
        logger.warning(f"Tesseract path from environment variable does not exist: {path}")

    # Then check common installation paths
    possible_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Tesseract-OCR\tesseract.exe',
        '/usr/bin/tesseract',
        '/usr/local/bin/tesseract',
        '/opt/homebrew/bin/tesseract'  # macOS Homebrew path
    ]
    
    # Try to find tesseract using which command
    try:
        import subprocess
        which_result = subprocess.run(['which', 'tesseract'], capture_output=True, text=True)
        if which_result.returncode == 0:
            path = which_result.stdout.strip()
            if os.path.exists(path):
                return path
    except Exception as e:
        logger.warning(f"Error running which command: {str(e)}")
    
    # Check all possible paths
    for path in possible_paths:
        if os.path.exists(path):
            return path
            
    return None

tesseract_path = get_tesseract_path()
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    logger.info("Tesseract found at: %s", tesseract_path)
else:
    logger.error("Tesseract not found. Please install Tesseract OCR")
    logger.error("After installation, make sure to add Tesseract to your system PATH")
    # Instead of raising an exception, we'll set a default path for Render
    if os.environ.get('RENDER'):
        tesseract_path = '/usr/bin/tesseract'
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        logger.info("Using default Tesseract path for Render: %s", tesseract_path)
    else:
        raise Exception("Tesseract OCR is not installed. Please install it first.")

def validate_image_data(image_data):
    if not image_data:
        raise ValueError("No image data provided")
    if not isinstance(image_data, str):
        raise ValueError("Image data must be a string")
    if not image_data.startswith('data:image/'):
        raise ValueError("Invalid image format. Expected base64 encoded image data")
    
    # Check image type
    image_type = image_data.split(';')[0].split(':')[1]
    if image_type not in ALLOWED_IMAGE_TYPES:
        raise ValueError(f"Unsupported image type. Allowed types: {', '.join(ALLOWED_IMAGE_TYPES)}")

def validate_image_size(image):
    width, height = image.size
    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise ValueError(f"Image dimensions too large. Maximum dimension is {MAX_IMAGE_DIMENSION}px")
    
    # Check file size
    buffer = io.BytesIO()
    image.save(buffer, format=image.format or 'JPEG')
    size = buffer.tell()
    if size > MAX_IMAGE_SIZE:
        raise ValueError(f"Image size too large. Maximum size is {MAX_IMAGE_SIZE/1024/1024}MB")

def process_image(image_data):
    if not tesseract_path:
        raise Exception("Tesseract OCR is not installed. Please install it first.")

    try:
        validate_image_data(image_data)
        
        # Convert base64 to image
        try:
            image_bytes = base64.b64decode(image_data.split(',')[1])
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            raise ValueError(f"Invalid image data: {str(e)}")
        
        # Validate image size and dimensions
        validate_image_size(image)
        
        # Convert PIL Image to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding for better text detection
        threshold = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Perform text detection with config
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(threshold, config=custom_config)
        
        if not text.strip():
            return "No text detected in the image."
            
        return text.strip()
        
    except Exception as e:
        logger.error("Error processing image: %s", str(e))
        raise Exception(f"Error processing image: {str(e)}")

@app.route('/api/detect-text', methods=['POST'])
@rate_limit
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
        
    except ValueError as e:
        logger.error("Validation error: %s", str(e))
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error("Error in detect_text endpoint: %s", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'tesseract_installed': bool(tesseract_path),
        'version': '1.0.0'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    logger.info("Starting Flask server on port %d (debug=%s)...", port, debug)
    app.run(debug=debug, port=port, host='0.0.0.0') 