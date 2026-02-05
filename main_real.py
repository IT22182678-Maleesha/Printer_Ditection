# #!/usr/bin/env python3
# """
# Main script for Printer Counter Detection System
# Run: python main.py [command]
# """

# import os
# import sys
# import argparse

# def main():
#     print("="*50)
#     print("PRINTER COUNTER DETECTION SYSTEM")
#     print("="*50)
    
#     parser = argparse.ArgumentParser(description='Control the detection system')
#     parser.add_argument('command', choices=['setup', 'prepare', 'train', 'convert', 'app', 'test'],
#                        help='Command to execute')
    
#     if len(sys.argv) < 2:
#         print("\nAvailable commands:")
#         print("  setup    - Install requirements and create folders")
#         print("  prepare  - Prepare dataset from PNG files")
#         print("  train    - Train the detection model")
#         print("  convert  - Convert to TFLite format")
#         print("  app      - Run web application")
#         print("  test     - Test system")
#         print("\nExample: python main.py setup")
#         return
    
#     command = sys.argv[1]
    
#     if command == 'setup':
#         print("\nRunning setup...")
#         os.system("python setup.py")
        
#     elif command == 'prepare':
#         print("\nPreparing data...")
#         # Check if images exist
#         if not os.path.exists('data/raw'):
#             print("ERROR: 'data/raw' folder not found!")
#             print("Please create 'data/raw' folder and put PNG images in it")
#             return
        
#         os.system("python scripts/prepare_data.py")
        
#     elif command == 'train':
#         print("\nTraining model...")
#         os.system("python scripts/train_model.py")
        
#     elif command == 'convert':
#         print("\nConverting to TFLite...")
#         os.system("python scripts/convert_to_tflite.py")
        
#     elif command == 'app':
#         print("\nStarting web application...")
#         print("Open browser to: http://localhost:8501")
#         os.system("streamlit run app.py")
        
#     elif command == 'test':
#         print("\nTesting system...")
#         # Quick test
#         import tensorflow as tf
#         import cv2
#         print(f"TensorFlow: {tf.__version__}")
#         print(f"OpenCV: {cv2.__version__}")
        
#         # Check folders
#         folders = ['data/raw', 'data/processed', 'models', 'scripts']
#         for folder in folders:
#             if os.path.exists(folder):
#                 print(f"✓ {folder} exists")
#             else:
#                 print(f"✗ {folder} missing")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Main script for Printer Counter Detection System
Run: python main.py [command]
"""

#!/usr/bin/env python3
"""
Main controller for Printer Counter Detection System
"""

#!/usr/bin/env python3
"""
Main script for Printer Counter Detection System
"""

#!/usr/bin/env python3
"""
Main controller for Printer Counter Detection System
"""

#!/usr/bin/env python3
"""
Main Controller - Simplified
"""

#!/usr/bin/env python3
"""
Main Controller for Printer Counter Detection
"""

#!/usr/bin/env python3
"""
Main script for Real Printer Counter Detection
"""
import os
import sys
import argparse

def main():
    print("="*60)
    print("REAL PRINTER COUNTER DETECTION SYSTEM")
    print("="*60)
    
    parser = argparse.ArgumentParser(description='Control the real detection system')
    parser.add_argument('command', choices=['setup', 'prepare', 'train', 'test', 'app', 'ocr'],
                       help='Command to execute')
    
    if len(sys.argv) < 2:
        print("\n📋 Available commands:")
        print("  setup    - Install all requirements")
        print("  prepare  - Prepare real dataset from your images")
        print("  train    - Train real AI model")
        print("  test     - Test system with sample image")
        print("  app      - Run web application")
        print("  ocr      - Try OCR on an image")
        print("\nExample: python main_real.py prepare")
        return
    
    command = sys.argv[1]
    
    if command == 'setup':
        print("\n🔧 Running setup...")
        # Install additional packages for OCR
        os.system("pip install easyocr pytesseract opencv-python")
        print("✓ Setup complete")
        
    elif command == 'prepare':
        print("\n📊 Preparing real dataset...")
        if not os.path.exists('data/raw'):
            print("ERROR: 'data/raw' folder not found!")
            print("Please create 'data/raw' and add your printer images")
            return
        
        # Count images
        images = [f for f in os.listdir('data/raw') if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Found {len(images)} images in data/raw/")
        
        os.system("python scripts/prepare_real_dataset.py")
        
    elif command == 'train':
        print("\n🤖 Training real model...")
        os.system("python scripts/train_real_model.py")
        
    elif command == 'test':
        print("\n🧪 Testing system...")
        # Test with sample
        if os.path.exists('test_sample.png'):
            print("Testing with sample image...")
            import cv2
            img = cv2.imread('test_sample.png')
            print(f"Image size: {img.shape}")
            
            # Try to load model
            try:
                import tensorflow as tf
                print(f"TensorFlow: {tf.__version__}")
                
                if os.path.exists('models/real_model/final_model.h5'):
                    print("✓ Model exists")
                else:
                    print("✗ Model not trained yet")
            except:
                print("TensorFlow not available")
        else:
            print("Create test sample first")
            os.system("python create_test_image.py")
        
    elif command == 'app':
        print("\n🌐 Starting web application...")
        print("Open browser to: http://localhost:8501")
        os.system("streamlit run app_real.py")
        
    elif command == 'ocr':
        print("\n🔤 Testing OCR on image...")
        if len(sys.argv) > 2:
            image_path = sys.argv[2]
            if os.path.exists(image_path):
                os.system(f"python test_ocr.py {image_path}")
            else:
                print(f"Image not found: {image_path}")
        else:
            print("Please specify image path:")
            print("python main_real.py ocr path/to/image.png")

if __name__ == "__main__":
    main()