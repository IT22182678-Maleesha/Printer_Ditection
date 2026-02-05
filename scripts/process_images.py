# import cv2
# import numpy as np
# import os
# import json
# from pathlib import Path
# import pytesseract
# from PIL import Image
# import re

# # Configure Tesseract path (update for your system)
# # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# class PrinterImageProcessor:
#     def __init__(self):
#         self.counter_patterns = {
#             '101': r'101\s*[:=]?\s*([0-9]+)',
#             '102': r'102\s*[:=]?\s*([0-9]+)',
#             '103': r'103\s*[:=]?\s*([0-9]+)',
#             '112': r'112\s*[:=]?\s*([0-9]+)',
#             '113': r'113\s*[:=]?\s*([0-9]+)',
#             '114': r'114\s*[:=]?\s*([0-9]+)',
#             '118': r'118\s*[:=]?\s*([0-9]+)',
#             '201': r'201\s*[:=]?\s*([0-9]+)',
#             '203': r'203\s*[:=]?\s*([0-9]+)',
#             '301': r'301\s*[:=]?\s*([0-9]+)',
#             '302': r'302\s*[:=]?\s*([0-9]+)',
#             '401': r'401\s*[:=]?\s*([0-9]+)',
#             '402': r'402\s*[:=]?\s*([0-9]+)',
#             '501': r'501\s*[:=]?\s*([0-9]+)',
#             '502': r'502\s*[:=]?\s*([0-9]+)',
#         }
        
#     def preprocess_image(self, image_path):
#         """Enhance image for better OCR"""
#         img = cv2.imread(image_path)
#         if img is None:
#             return None
            
#         # Convert to grayscale
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
#         # Increase contrast
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#         enhanced = clahe.apply(gray)
        
#         # Denoise
#         denoised = cv2.medianBlur(enhanced, 3)
        
#         # Threshold
#         _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
#         # Deskew if needed
#         coords = np.column_stack(np.where(thresh > 0))
#         angle = cv2.minAreaRect(coords)[-1]
#         if angle < -45:
#             angle = -(90 + angle)
#         else:
#             angle = -angle
            
#         (h, w) = thresh.shape[:2]
#         center = (w // 2, h // 2)
#         M = cv2.getRotationMatrix2D(center, angle, 1.0)
#         rotated = cv2.warpAffine(thresh, M, (w, h), 
#                                 flags=cv2.INTER_CUBIC, 
#                                 borderMode=cv2.BORDER_REPLICATE)
        
#         return rotated
    
#     def extract_text(self, image_path):
#         """Extract text from image using OCR"""
#         processed_img = self.preprocess_image(image_path)
#         if processed_img is None:
#             return ""
        
#         # Use Tesseract with custom configuration
#         custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789:()abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\s'
#         text = pytesseract.image_to_string(processed_img, config=custom_config)
#         return text
    
#     def extract_counters(self, image_path):
#         """Extract all counter values from image"""
#         text = self.extract_text(image_path)
#         counters = {}
        
#         for counter_id, pattern in self.counter_patterns.items():
#             matches = re.findall(pattern, text, re.IGNORECASE)
#             if matches:
#                 # Take the first match (most likely)
#                 counters[counter_id] = int(matches[0])
        
#         return counters
    
#     def process_folder(self, input_folder, output_file):
#         """Process all images in folder and save results"""
#         results = {}
        
#         for img_file in os.listdir(input_folder):
#             if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
#                 img_path = os.path.join(input_folder, img_file)
#                 print(f"Processing: {img_file}")
                
#                 try:
#                     counters = self.extract_counters(img_path)
#                     if counters:
#                         results[img_file] = {
#                             'counters': counters,
#                             'total_count': sum(counters.values())
#                         }
#                         print(f"  Found: {counters}")
#                     else:
#                         print(f"  No counters found")
#                 except Exception as e:
#                     print(f"  Error: {e}")
        
#         # Save results
#         with open(output_file, 'w') as f:
#             json.dump(results, f, indent=4)
        
#         print(f"\nProcessed {len(results)} images")
#         print(f"Results saved to {output_file}")
        
#         return results

# if __name__ == "__main__":
#     processor = PrinterImageProcessor()
    
#     # Process your images
#     input_folder = "data/raw"
#     output_file = "data/labels/extracted_counters.json"
    
#     if not os.path.exists(input_folder):
#         os.makedirs(input_folder, exist_ok=True)
#         print(f"Please put your images in {input_folder}")
#     else:
#         results = processor.process_folder(input_folder, output_file)