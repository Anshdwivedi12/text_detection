# Image to Text Converter

A web application that extracts text from images using OCR (Optical Character Recognition) technology. Built with React and Flask.

## Features

- Upload images and extract text
- Support for multiple image formats (JPG, PNG, TIFF, BMP)
- Real-time text detection
- Copy extracted text to clipboard
- Responsive design
- Drag and drop support

## Tech Stack

- Frontend: React.js, Framer Motion
- Backend: Flask, OpenCV, Tesseract OCR
- Deployment: Render.com

## Prerequisites

- Python 3.9+
- Node.js 14+
- Tesseract OCR

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Anshdwivedi12/image-to-text.git
cd image-to-text
```

2. Install backend dependencies:
```bash
pip install -r requirements.txt
```

3. Install frontend dependencies:
```bash
cd frontend
npm install
```

4. Install Tesseract OCR:
- Windows: Download and install from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
- Linux: `sudo apt-get install tesseract-ocr`
- Mac: `brew install tesseract`

## Running the Application

1. Start the backend server:
```bash
python app.py
```

2. Start the frontend development server:
```bash
cd frontend
npm start
```

The application will be available at `http://localhost:3000`

## Deployment

The application is configured for deployment on Render.com. Follow these steps:

1. Create a Render account
2. Deploy the backend as a Web Service
3. Deploy the frontend as a Static Site
4. Configure environment variables

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Ansh Dwivedi
- LinkedIn: [Ansh Dwivedi](https://www.linkedin.com/in/anshdwivedi-/)
- GitHub: [Anshdwivedi12](https://github.com/Anshdwivedi12) 