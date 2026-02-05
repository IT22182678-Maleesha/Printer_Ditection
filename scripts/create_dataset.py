# import os
# import json
# import cv2
# import numpy as np
# from sklearn.model_selection import train_test_split
# import shutil

# class DatasetCreator:
#     def __init__(self):
#         self.image_size = (224, 224)
        
#     def create_ocr_dataset(self, image_folder, labels_file, output_folder):
#         """Create dataset for OCR training"""
#         with open(labels_file, 'r') as f:
#             labels = json.load(f)
        
#         # Create output directories
#         os.makedirs(os.path.join(output_folder, 'train', 'images'), exist_ok=True)
#         os.makedirs(os.path.join(output_folder, 'train', 'labels'), exist_ok=True)
#         os.makedirs(os.path.join(output_folder, 'val', 'images'), exist_ok=True)
#         os.makedirs(os.path.join(output_folder, 'val', 'labels'), exist_ok=True)
#         os.makedirs(os.path.join(output_folder, 'test', 'images'), exist_ok=True)
#         os.makedirs(os.path.join(output_folder, 'test', 'labels'), exist_ok=True)
        
#         image_files = []
#         all_labels = []
        
#         for img_file in os.listdir(image_folder):
#             if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
#                 if img_file in labels:
#                     image_files.append(img_file)
#                     all_labels.append(labels[img_file])
        
#         # Split data
#         train_files, test_files, train_labels, test_labels = train_test_split(
#             image_files, all_labels, test_size=0.2, random_state=42
#         )
#         train_files, val_files, train_labels, val_labels = train_test_split(
#             train_files, train_labels, test_size=0.125, random_state=42
#         )
        
#         # Process and save datasets
#         self._process_split(train_files, train_labels, image_folder, 
#                           output_folder, 'train')
#         self._process_split(val_files, val_labels, image_folder, 
#                           output_folder, 'val')
#         self._process_split(test_files, test_labels, image_folder, 
#                           output_folder, 'test')
        
#         # Save metadata
#         metadata = {
#             'num_train': len(train_files),
#             'num_val': len(val_files),
#             'num_test': len(test_files),
#             'image_size': self.image_size,
#             'counter_types': ['101', '102', '103', '112', '113', '114', '118', 
#                             '201', '203', '301', '302', '401', '402', '501', '502']
#         }
        
#         with open(os.path.join(output_folder, 'metadata.json'), 'w') as f:
#             json.dump(metadata, f, indent=4)
        
#         print(f"Dataset created:")
#         print(f"  Training: {len(train_files)} images")
#         print(f"  Validation: {len(val_files)} images")
#         print(f"  Test: {len(test_files)} images")
        
#         return metadata
    
#     def _process_split(self, files, labels, image_folder, output_folder, split_name):
#         """Process and save a data split"""
#         for i, (img_file, label_data) in enumerate(zip(files, labels)):
#             # Load image
#             img_path = os.path.join(image_folder, img_file)
#             img = cv2.imread(img_path)
            
#             if img is not None:
#                 # Resize and normalize
#                 img = cv2.resize(img, self.image_size)
#                 img = img.astype(np.float32) / 255.0
                
#                 # Save image
#                 img_filename = f"{split_name}_{i:04d}.npy"
#                 img_save_path = os.path.join(output_folder, split_name, 'images', img_filename)
#                 np.save(img_save_path, img)
                
#                 # Save label
#                 label_filename = f"{split_name}_{i:04d}.json"
#                 label_save_path = os.path.join(output_folder, split_name, 'labels', label_filename)
                
#                 with open(label_save_path, 'w') as f:
#                     json.dump(label_data, f, indent=4)
                
#                 if (i + 1) % 10 == 0:
#                     print(f"  Processed {i + 1}/{len(files)} for {split_name}")

# if __name__ == "__main__":
#     creator = DatasetCreator()
    
#     # Create dataset from your labeled images
#     image_folder = "data/raw"
#     labels_file = "data/labels/extracted_counters.json"
#     output_folder = "data/processed"
    
#     if not os.path.exists(labels_file):
#         print(f"Please run process_images.py first to create {labels_file}")
#     else:
#         metadata = creator.create_ocr_dataset(image_folder, labels_file, output_folder)