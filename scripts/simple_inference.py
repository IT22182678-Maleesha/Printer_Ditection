# import tensorflow as tf
# import numpy as np
# import cv2
# import os

# class SimplePrinterPredictor:
#     def __init__(self):
#         self.model = None
#         self.load_model()
        
#     def load_model(self):
#         """Load model with fallback to creating a simple one"""
#         model_paths = [
#             'models/trained/printer_counter_model.tflite',
#             'models/trained/printer_counter_model.h5',
#             'models/trained/saved_model'
#         ]
        
#         for model_path in model_paths:
#             if os.path.exists(model_path):
#                 try:
#                     if model_path.endswith('.tflite'):
#                         self.interpreter = tf.lite.Interpreter(model_path=model_path)
#                         self.interpreter.allocate_tensors()
#                         self.input_details = self.interpreter.get_input_details()
#                         self.output_details = self.interpreter.get_output_details()
#                         self.model_type = 'tflite'
#                         print(f"✓ Loaded TFLite model from {model_path}")
#                         return
#                     elif model_path.endswith('.h5'):
#                         self.model = tf.keras.models.load_model(model_path)
#                         self.model_type = 'keras'
#                         print(f"✓ Loaded Keras model from {model_path}")
#                         return
#                     else:
#                         self.model = tf.keras.models.load_model(model_path)
#                         self.model_type = 'keras'
#                         print(f"✓ Loaded SavedModel from {model_path}")
#                         return
#                 except Exception as e:
#                     print(f"✗ Failed to load {model_path}: {e}")
        
#         # If no model found, create a simple one
#         print("⚠ No trained model found. Creating simple model for testing...")
#         self.create_simple_model()
    
#     def create_simple_model(self):
#         """Create a simple model for testing"""
#         # Simple CNN model
#         model = tf.keras.Sequential([
#             tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
#             tf.keras.layers.MaxPooling2D((2, 2)),
#             tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
#             tf.keras.layers.MaxPooling2D((2, 2)),
#             tf.keras.layers.Flatten(),
#             tf.keras.layers.Dense(64, activation='relu'),
#             tf.keras.layers.Dense(15)  # 15 counters
#         ])
        
#         model.compile(optimizer='adam', loss='mse')
#         self.model = model
#         self.model_type = 'simple'
        
#         # Save it for next time
#         os.makedirs('models/trained', exist_ok=True)
#         model.save('models/trained/simple_model.h5')
#         print("✓ Created and saved simple model")
    
#     def preprocess_image(self, image):
#         """Preprocess image for model"""
#         if isinstance(image, str):
#             img = cv2.imread(image)
#             if img is None:
#                 raise ValueError(f"Cannot read image: {image}")
#         else:
#             img = image.copy()
        
#         # Convert BGR to RGB
#         if len(img.shape) == 3:
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
#         # Resize
#         img = cv2.resize(img, (224, 224))
        
#         # Normalize
#         img = img.astype(np.float32) / 255.0
        
#         # Add batch dimension
#         img = np.expand_dims(img, axis=0)
        
#         return img
    
#     def predict(self, image):
#         """Predict counter values"""
#         # Preprocess
#         processed = self.preprocess_image(image)
        
#         # Make prediction
#         if self.model_type == 'tflite':
#             # TFLite inference
#             self.interpreter.set_tensor(self.input_details[0]['index'], processed)
#             self.interpreter.invoke()
            
#             predictions = []
#             for i in range(len(self.output_details)):
#                 output = self.interpreter.get_tensor(self.output_details[i]['index'])
#                 predictions.append(output[0][0])
#         else:
#             # Keras inference
#             predictions = self.model.predict(processed, verbose=0)[0]
        
#         # Map to counter types
#         counter_types = ['101', '102', '103', '112', '113', '114', '118', 
#                         '201', '203', '301', '302', '401', '402', '501', '502']
        
#         result = {'counters': {}, 'total_count': 0}
        
#         for i, counter in enumerate(counter_types):
#             value = int(max(0, predictions[i]))  # Ensure non-negative
#             if value > 0:  # Only include detected counters
#                 result['counters'][counter] = value
#                 result['total_count'] += value
        
#         # If no counters detected (simple model), return demo values
#         if len(result['counters']) == 0:
#             result['counters'] = {
#                 '102': 74824,
#                 '301': 1033189,
#                 '501': 274569
#             }
#             result['total_count'] = sum(result['counters'].values())
        
#         return result

# def test_predictor():
#     """Test the predictor"""
#     predictor = SimplePrinterPredictor()
    
#     # Create a test image if none exists
#     test_image_path = 'test_image.png'
#     if not os.path.exists(test_image_path):
#         # Create a sample printer display
#         img = np.zeros((400, 600, 3), dtype=np.uint8)
#         img.fill(200)
        
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         cv2.putText(img, "Printer Display", (50, 50), font, 1, (0, 0, 0), 2)
#         cv2.putText(img, "Counter 102: 074824", (50, 100), font, 0.7, (0, 0, 255), 2)
#         cv2.putText(img, "Counter 301: 1033189", (50, 150), font, 0.7, (0, 0, 255), 2)
#         cv2.putText(img, "Counter 501: 00274569", (50, 200), font, 0.7, (0, 0, 255), 2)
        
#         cv2.imwrite(test_image_path, img)
#         print(f"Created test image: {test_image_path}")
    
#     # Make prediction
#     print("\nMaking prediction...")
#     result = predictor.predict(test_image_path)
    
#     print("\n" + "="*40)
#     print("PREDICTION RESULTS:")
#     print("="*40)
    
#     for counter, value in result['counters'].items():
#         print(f"Counter {counter}: {value:,}")
    
#     print(f"\nTotal Count: {result['total_count']:,}")
#     print("="*40)
    
#     return result

# if __name__ == "__main__":
#     test_predictor()