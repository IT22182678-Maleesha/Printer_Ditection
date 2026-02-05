🖨️ Printer Counter Detection System

📋 Overview

An AI-powered system to automatically detect and extract printer counter values from digital display images. This system uses machine learning to recognize various printer counter types (102, 301, 502, etc.) and their corresponding values.

✨ Features
Image Upload: Upload printer display images in PNG, JPG, JPEG, or BMP formats

Automatic Detection: AI model detects counter regions and extracts values

Manual Labeling: GUI tool for manually labeling training data

Model Training: Train custom TensorFlow models on your specific printer images

Export Results: Export detected values as JSON or CSV

Web Interface: Streamlit web app for easy interaction

Multi-Platform: Works on macOS, Windows, and Linux

🚀 Quick Start
# Clone or create project directory
mkdir printer-detection
cd printer-detection

# Create virtual environment (macOS/Linux)
python3 -m venv venv
source venv/bin/activate

# Create directory structure
mkdir -p data/raw_images
mkdir -p data/annotations
mkdir -p models
mkdir -p scripts
mkdir -p test_images

2. Install Dependencies
   # Install basic packages
pip install Pillow==10.1.0
pip install numpy==1.24.0
pip install opencv-python==4.9.0.80
pip install matplotlib==3.7.2

# Install TensorFlow (macOS)
pip install tensorflow-macos==2.15.0

# For Apple Silicon (M1/M2/M3) add:
pip install tensorflow-metal==1.1.0

# Install additional packages
pip install pandas==2.1.3
pip install scikit-learn==1.3.2
pip install streamlit==1.28.1

printer-detection/
├── data/
│   ├── raw_images/          # Your original printer images
│   └── labels.json          # Manual labels for training
├── models/                  # Trained models
├── scripts/                 # Python scripts
│   ├── simple_labeling.py   # GUI for manual labeling
│   ├── simple_training.py   # Model training script
│   ├── simple_prediction.py # Prediction script
│   └── analyze_images.py    # Image analysis tool
├── simple_app.py            # Streamlit web app
├── requirements.txt         # Dependencies
└── README.md               # This file

🔄 Workflow Summary
Data Collection → Gather printer display images

Labeling → Manually enter counter values (optional)

Training → Train AI model on your data

Prediction → Detect counters in new images

Export → Save results for reporting

🤝 Contributing
Feel free to:

Report bugs or issues

Suggest new features

Submit pull requests

Share your labeled datasets
