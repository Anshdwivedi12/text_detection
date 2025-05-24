import requests
import json
import base64
from PIL import Image
import io

# Replace with your Render deployment URL
BASE_URL = "https://text-detection-backend.onrender.com"

def test_health():
    print("\nTesting Health Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error testing health endpoint: {str(e)}")
        return False

def test_text_detection():
    print("\nTesting Text Detection Endpoint...")
    try:
        # Create a simple test image with text
        img = Image.new('RGB', (200, 50), color='white')
        img.save('test.png')
        
        # Convert image to base64
        with open('test.png', 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Prepare the request
        headers = {'Content-Type': 'application/json'}
        data = {
            'image': f'data:image/png;base64,{encoded_string}'
        }
        
        # Make the request
        response = requests.post(
            f"{BASE_URL}/api/detect-text",
            headers=headers,
            json=data
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error testing text detection endpoint: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting deployment tests...")
    
    # Test health endpoint
    health_ok = test_health()
    print(f"Health Check {'PASSED' if health_ok else 'FAILED'}")
    
    # Test text detection endpoint
    detection_ok = test_text_detection()
    print(f"Text Detection {'PASSED' if detection_ok else 'FAILED'}")
    
    # Overall status
    if health_ok and detection_ok:
        print("\n✅ All tests passed! Deployment is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the deployment logs.") 