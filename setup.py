# #!/usr/bin/env python3
# """
# Setup script for Printer Counter Detection System
# """

# import subprocess
# import sys
# import os

# def check_and_install():
#     """Check and install required packages"""
#     print("="*50)
#     print("SETTING UP PRINTER COUNTER DETECTION SYSTEM")
#     print("="*50)
    
#     # List of required packages
#     requirements = [
#         'tensorflow>=2.14.0',
#         'opencv-python>=4.8.0',
#         'pillow>=10.0.0',
#         'numpy>=1.24.0',
#         'matplotlib>=3.7.0',
#         'pandas>=2.0.0',
#         'scikit-learn>=1.3.0',
#         'streamlit>=1.28.0',
#         'tqdm>=4.66.0',
#     ]
    
#     print("\n1. Installing packages...")
#     for package in requirements:
#         try:
#             subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
#             print(f"   ✓ {package}")
#         except:
#             print(f"   ✗ {package}")
    
#     # Create necessary folders
#     print("\n2. Creating folder structure...")
#     folders = [
#         'data/raw',
#         'data/processed/train/images',
#         'data/processed/train/labels',
#         'data/processed/val/images',
#         'data/processed/val/labels',
#         'data/processed/test/images',
#         'data/processed/test/labels',
#         'models/checkpoints',
#         'models/saved_model',
#         'models/tflite',
#         'scripts',
#         'app',
#         'samples'
#     ]
    
#     for folder in folders:
#         os.makedirs(folder, exist_ok=True)
#         print(f"   ✓ {folder}")
    
#     print("\n" + "="*50)
#     print("SETUP COMPLETED SUCCESSFULLY!")
#     print("="*50)
    
#     print("\nNEXT STEPS:")
#     print("1. Put your PNG images in 'data/raw/' folder")
#     print("2. Run: python scripts/prepare_data.py")
#     print("3. Run: python scripts/train_model.py")
#     print("4. Run: streamlit run app.py")

# if __name__ == "__main__":
#     check_and_install()


#!/usr/bin/env python3
"""
Setup script for Printer Counter Detection System
"""

#!/usr/bin/env python3
"""
Setup script for Printer Counter Detection System
"""

#!/usr/bin/env python3
"""
Complete setup script for Printer Counter Detection System
"""

#!/usr/bin/env python3
"""
Simplified setup script
"""

#!/usr/bin/env python3
"""
Setup script for Printer Counter Detection System
"""

import os
import subprocess
import sys
import json

def setup_system():
    print("="*60)
    print("PRINTER COUNTER DETECTION SYSTEM - SETUP")
    print("="*60)
    
    # Create directory structure
    directories = [
        'data/raw',
        'data/processed',
        'data/labels',
        'models/trained',
        'models/checkpoints',
        'models/tflite',
        'scripts',
        'logs'
    ]
    
    print("\n1. Creating directory structure...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✓ Created {directory}")
    
    # Install requirements
    print("\n2. Installing dependencies...")
    
    requirements = [
        'tensorflow>=2.14.0',
        'opencv-python>=4.8.0',
        'pillow>=10.0.0',
        'numpy>=1.24.0',
        'streamlit>=1.28.0',
        'matplotlib>=3.7.0',
        'scikit-learn>=1.3.0',
        'pandas>=2.0.0',
        'tqdm>=4.66.0',
        'pytesseract>=0.3.10'
    ]
    
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"   ✓ Installed {package}")
        except:
            print(f"   ✗ Failed to install {package}")
    
    # Create sample test image
    print("\n3. Creating sample test images...")
    create_sample_images()
    
    # Create empty labels file for now
    print("\n4. Creating initial label structure...")
    initial_labels = {
        "counter_types": ["101", "102", "103", "112", "113", "114", "118", 
                         "201", "203", "301", "302", "401", "402", "501", "502"]
    }
    
    with open('data/labels/counter_types.json', 'w') as f:
        json.dump(initial_labels, f, indent=4)
    
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    
    print("\nNEXT STEPS:")
    print("1. Put your PNG images in 'data/raw/' folder")
    print("2. Run: python scripts/process_images.py")
    print("3. Run: python scripts/train_model.py")
    print("4. Run: streamlit run app.py")

def create_sample_images():
    """Create sample printer display images for testing"""
    import cv2
    import numpy as np
    
    samples = [
        {
            'name': 'sample_1.png',
            'counters': {
                '101': 12345,
                '102': 67890,
                '301': 54321,
                '501': 9876
            }
        },
        {
            'name': 'sample_2.png',
            'counters': {
                '102': 55555,
                '201': 33333,
                '302': 77777,
                '401': 22222
            }
        }
    ]
    
    for sample in samples:
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        img.fill(240)  # Light gray background
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_pos = 50
        
        # Title
        cv2.putText(img, "Printer Counter Display", (50, y_pos), font, 0.8, (0, 0, 0), 2)
        y_pos += 50
        
        # Counters
        for counter, value in sample['counters'].items():
            text = f"Counter {counter}: {value:08d}"
            cv2.putText(img, text, (80, y_pos), font, 0.7, (0, 100, 200), 2)
            y_pos += 40
        
        # Save
        cv2.imwrite(os.path.join('data/raw', sample['name']), img)
        print(f"   Created sample: {sample['name']}")

if __name__ == "__main__":
    setup_system()