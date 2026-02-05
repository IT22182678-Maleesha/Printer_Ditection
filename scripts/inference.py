# import tensorflow as tf
# import numpy as np
# import cv2
# import json
# from pathlib import Path

# class PrinterCounterPredictor:
#     def __init__(self, model_path='models/trained/printer_counter_model.tflite'):
#         self.model_path = model_path
#         self.image_size = (224, 224)
        
#         # Load TFLite model
#         self.interpreter = tf.lite.Interpreter(model_path=model_path)
#         self.interpreter.allocate_tensors()
        
#         # Get input/output details
#         self.input_details = self.interpreter.get_input_details()
#         self.output_details = self.interpreter.get_output_details()
        
#         # Counter names in order
#         self.counter_names = ['101', '102', '103', '112', '113', '114', '118', 
#                              '201', '203', '301', '302', '401', '402', '501', '502']
    
#     def preprocess_image(self, image_path):
#         """Preprocess image for model inference"""
#         # Load image
#         if isinstance(image_path, str):
#             img = cv2.imread(image_path)
#         else:
#             img = image_path
        
#         if img is None:
#             raise ValueError("Could not load image")
        
#         # Convert BGR to RGB
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
#         # Resize
#         img = cv2.resize(img, self.image_size)
        
#         # Normalize
#         img = img.astype(np.float32) / 255.0
        
#         # Add batch dimension
#         img = np.expand_dims(img, axis=0)
        
#         return img
    
#     def predict(self, image_path):
#         """Predict counter values from image"""
#         # Preprocess image
#         input_data = self.preprocess_image(image_path)
        
#         # Set input tensor
#         self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
#         # Run inference
#         self.interpreter.invoke()
        
#         # Get predictions
#         predictions = {}
#         total_count = 0
        
#         for i, counter_name in enumerate(self.counter_names):
#             output_data = self.interpreter.get_tensor(self.output_details[i]['index'])
#             counter_value = int(max(0, output_data[0][0]))  # Ensure non-negative
            
#             if counter_value > 0:  # Only include non-zero counters
#                 predictions[counter_name] = counter_value
#                 total_count += counter_value
        
#         return {
#             'counters': predictions,
#             'total_count': total_count
#         }
    
#     def predict_from_camera(self, camera_index=0):
#         """Real-time prediction from camera"""
#         cap = cv2.VideoCapture(camera_index)
        
#         print("Press 'q' to quit, 'c' to capture and predict")
        
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             # Display frame
#             display_frame = frame.copy()
#             cv2.putText(display_frame, "Press 'c' to predict, 'q' to quit", 
#                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
#             cv2.imshow('Printer Counter Detection', display_frame)
            
#             key = cv2.waitKey(1) & 0xFF
            
#             if key == ord('q'):
#                 break
#             elif key == ord('c'):
#                 # Predict on current frame
#                 try:
#                     result = self.predict(frame)
                    
#                     # Display results
#                     print("\n" + "="*40)
#                     print("PREDICTION RESULTS:")
#                     print("="*40)
                    
#                     for counter, value in result['counters'].items():
#                         print(f"Counter {counter}: {value}")
                    
#                     print(f"\nTotal Count: {result['total_count']}")
#                     print("="*40)
                    
#                     # Show results on image
#                     result_frame = frame.copy()
#                     y_pos = 50
                    
#                     for counter, value in list(result['counters'].items())[:5]:  # Show first 5
#                         cv2.putText(result_frame, f"Counter {counter}: {value}", 
#                                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#                         y_pos += 30
                    
#                     cv2.putText(result_frame, f"Total: {result['total_count']}", 
#                                (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    
#                     cv2.imshow('Prediction Results', result_frame)
                    
#                 except Exception as e:
#                     print(f"Prediction error: {e}")
        
#         cap.release()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     # Create predictor
#     predictor = PrinterCounterPredictor()
    
#     # Test with sample image
#     test_image = "test_sample.png"
    
#     if Path(test_image).exists():
#         print(f"Testing with {test_image}...")
#         result = predictor.predict(test_image)
        
#         print("\n" + "="*50)
#         print("PREDICTION RESULTS:")
#         print("="*50)
        
#         for counter, value in result['counters'].items():
#             print(f"Counter {counter}: {value}")
        
#         print(f"\nTotal Count: {result['total_count']}")
#         print("="*50)
#     else:
#         print("No test image found. Starting camera prediction...")
#         predictor.predict_from_camera()




# scripts/simple_inference.py
import numpy as np
import cv2
import os

class SimpleCounterDetector:
    def __init__(self):
        self.image_size = (224, 224)
        
    def preprocess(self, image):
        """Simple image preprocessing"""
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image.copy()
        
        # Resize
        img = cv2.resize(img, self.image_size)
        # Normalize
        img = img.astype(np.float32) / 255.0
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def predict(self, image):
        """Simple prediction - in real use, this would be your trained model"""
        # For demo, return values based on image analysis
        processed = self.preprocess(image)
        
        # Simple analysis
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image
            
        # Analyze image characteristics
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        # Generate values based on brightness
        base_value = int(brightness * 1000)
        
        # Return demo values
        return {
            '102': base_value,
            '301': base_value * 10 if brightness > 100 else 0,
            'total': base_value + (base_value * 10 if brightness > 100 else 0)
        }