# import tensorflow as tf
# from tensorflow import keras
# import numpy as np
# import os
# import json
# import cv2
# from datetime import datetime
# from tensorflow.keras import layers, models # type: ignore
# import matplotlib.pyplot as plt

# print("="*60)
# print("PRINTER COUNTER OCR MODEL TRAINING")
# print("="*60)

# class PrinterCounterModel:
#     def __init__(self, num_counter_types=15):
#         self.num_counter_types = num_counter_types
#         self.image_size = (224, 224)
        
#     def build_model(self):
#         """Build CNN model for counter value prediction"""
        
#         # Input layer
#         inputs = keras.Input(shape=(*self.image_size, 3))
        
#         # Feature extraction
#         x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
#         x = layers.MaxPooling2D((2, 2))(x)
#         x = layers.BatchNormalization()(x)
        
#         x = layers.Conv2D(64, (3, 3), activation='relu')(x)
#         x = layers.MaxPooling2D((2, 2))(x)
#         x = layers.BatchNormalization()(x)
        
#         x = layers.Conv2D(128, (3, 3), activation='relu')(x)
#         x = layers.MaxPooling2D((2, 2))(x)
#         x = layers.BatchNormalization()(x)
        
#         # Global features
#         x = layers.GlobalAveragePooling2D()(x)
#         x = layers.Dropout(0.5)(x)
        
#         # Multiple output heads for different counters
#         outputs = []
#         counter_names = ['101', '102', '103', '112', '113', '114', '118', 
#                         '201', '203', '301', '302', '401', '402', '501', '502']
        
#         for i in range(self.num_counter_types):
#             # Each counter has its own regression output
#             counter_output = layers.Dense(64, activation='relu')(x)
#             counter_output = layers.Dropout(0.3)(counter_output)
#             counter_output = layers.Dense(1, activation='linear', 
#                                          name=f'counter_{counter_names[i]}')(counter_output)
#             outputs.append(counter_output)
        
#         # Create model
#         model = keras.Model(inputs=inputs, outputs=outputs)
        
#         return model
    
#     def load_dataset(self, data_folder, split='train'):
#         """Load dataset from processed files"""
#         images_folder = os.path.join(data_folder, split, 'images')
#         labels_folder = os.path.join(data_folder, split, 'labels')
        
#         image_files = sorted([f for f in os.listdir(images_folder) if f.endswith('.npy')])
        
#         images = []
#         labels = []
        
#         for img_file in image_files:
#             # Load image
#             img_path = os.path.join(images_folder, img_file)
#             img = np.load(img_path)
#             images.append(img)
            
#             # Load corresponding label
#             label_file = img_file.replace('.npy', '.json')
#             label_path = os.path.join(labels_folder, label_file)
            
#             with open(label_path, 'r') as f:
#                 label_data = json.load(f)
            
#             # Extract counter values in fixed order
#             counter_values = []
#             counter_order = ['101', '102', '103', '112', '113', '114', '118', 
#                            '201', '203', '301', '302', '401', '402', '501', '502']
            
#             for counter in counter_order:
#                 if counter in label_data.get('counters', {}):
#                     counter_values.append(label_data['counters'][counter])
#                 else:
#                     counter_values.append(0.0)  # No value for this counter
            
#             labels.append(counter_values)
        
#         return np.array(images), np.array(labels)
    
#     def train(self, train_data, val_data, epochs=50):
#         """Train the model"""
#         train_images, train_labels = train_data
#         val_images, val_labels = val_data
        
#         # Build model
#         model = self.build_model()
        
#         # Compile with multiple outputs
#         model.compile(
#             optimizer=keras.optimizers.Adam(learning_rate=0.001),
#             loss={f'counter_{i}': 'mse' for i in 
#                  ['101', '102', '103', '112', '113', '114', '118', 
#                   '201', '203', '301', '302', '401', '402', '501', '502']},
#             metrics={f'counter_{i}': 'mae' for i in 
#                     ['101', '102', '103', '112', '113', '114', '118', 
#                      '201', '203', '301', '302', '401', '402', '501', '502']},
#         )
        
#         # Callbacks
#         callbacks = [
#             keras.callbacks.EarlyStopping(
#                 monitor='val_loss',
#                 patience=10,
#                 restore_best_weights=True
#             ),
#             keras.callbacks.ReduceLROnPlateau(
#                 monitor='val_loss',
#                 factor=0.5,
#                 patience=5,
#                 min_lr=1e-6
#             ),
#             keras.callbacks.ModelCheckpoint(
#                 filepath='models/checkpoints/best_model.h5',
#                 monitor='val_loss',
#                 save_best_only=True
#             ),
#             keras.callbacks.TensorBoard(
#                 log_dir=f'logs/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
#             )
#         ]
        
#         print(f"\nTraining on {len(train_images)} samples")
#         print(f"Validating on {len(val_images)} samples")
        
#         # Train
#         history = model.fit(
#             train_images,
#             {f'counter_{i}': train_labels[:, idx] for idx, i in enumerate(
#                 ['101', '102', '103', '112', '113', '114', '118', 
#                  '201', '203', '301', '302', '401', '402', '501', '502'])},
#             validation_data=(
#                 val_images,
#                 {f'counter_{i}': val_labels[:, idx] for idx, i in enumerate(
#                     ['101', '102', '103', '112', '113', '114', '118', 
#                      '201', '203', '301', '302', '401', '402', '501', '502'])}
#             ),
#             epochs=epochs,
#             batch_size=16,
#             callbacks=callbacks,
#             verbose=1
#         )
        
#         return model, history

# def main():
#     # Create model instance
#     model_trainer = PrinterCounterModel()
    
#     # Load datasets
#     print("\n1. Loading datasets...")
#     try:
#         train_images, train_labels = model_trainer.load_dataset('data/processed', 'train')
#         val_images, val_labels = model_trainer.load_dataset('data/processed', 'val')
#         test_images, test_labels = model_trainer.load_dataset('data/processed', 'test')
        
#         print(f"   Training set: {train_images.shape[0]} images")
#         print(f"   Validation set: {val_images.shape[0]} images")
#         print(f"   Test set: {test_images.shape[0]} images")
        
#     except Exception as e:
#         print(f"   Error loading dataset: {e}")
#         print("   Please run create_dataset.py first")
#         return
    
#     # Train model
#     print("\n2. Training model...")
#     model, history = model_trainer.train(
#         train_data=(train_images, train_labels),
#         val_data=(val_images, val_labels),
#         epochs=50
#     )
    
#     # Evaluate on test set
#     print("\n3. Evaluating model...")
#     test_results = model.evaluate(
#         test_images,
#         {f'counter_{i}': test_labels[:, idx] for idx, i in enumerate(
#             ['101', '102', '103', '112', '113', '114', '118', 
#              '201', '203', '301', '302', '401', '402', '501', '502'])},
#         verbose=0
#     )
    
#     print(f"   Test Loss: {test_results[0]:.4f}")
    
#     # Save model
#     print("\n4. Saving model...")
#     os.makedirs('models/trained', exist_ok=True)
#     model.save('models/trained/printer_counter_model.h5')
    
#     # Convert to TFLite
#     converter = tf.lite.TFLiteConverter.from_keras_model(model)
#     converter.optimizations = [tf.lite.Optimize.DEFAULT]
#     tflite_model = converter.convert()
    
#     with open('models/trained/printer_counter_model.tflite', 'wb') as f:
#         f.write(tflite_model)
    
#     print("   ✓ Model saved: models/trained/printer_counter_model.h5")
#     print("   ✓ TFLite model saved: models/trained/printer_counter_model.tflite")
    
#     # Plot training history
#     plt.figure(figsize=(12, 4))
    
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['loss'], label='Training Loss')
#     plt.plot(history.history['val_loss'], label='Validation Loss')
#     plt.title('Model Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
    
#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['counter_101_mae'], label='Counter 101 MAE')
#     plt.plot(history.history['counter_102_mae'], label='Counter 102 MAE')
#     plt.title('Mean Absolute Error')
#     plt.xlabel('Epoch')
#     plt.ylabel('MAE')
#     plt.legend()
    
#     plt.tight_layout()
#     plt.savefig('models/training_history.png')
    
#     print("\n" + "="*60)
#     print("TRAINING COMPLETE!")
#     print("="*60)

# if __name__ == "__main__":
#     main()


# scripts/auto_train_model.py - FIXED VERSION
import numpy as np
import os
import json

# Simple model training without TensorFlow for now
def create_simple_model():
    """Create simple model weights"""
    # For demo, create random weights
    # In production, use TensorFlow
    input_shape = (224, 224, 3)
    output_size = 15  # 15 counters
    
    # Simple weights (random initialization)
    weights = {
        'input_shape': input_shape,
        'output_size': output_size,
        'counters': ['101', '102', '103', '112', '113', '114', '118',
                    '201', '203', '301', '302', '401', '402', '501', '502']
    }
    
    return weights

def main():
    print("="*60)
    print("SIMPLE MODEL SETUP")
    print("="*60)
    
    # Check if dataset exists
    data_folder = 'data/auto_processed'
    
    if not os.path.exists(data_folder):
        print("❌ Dataset not found. Run auto_data_preparation.py first.")
        return
    
    # Load dataset info
    metadata_file = os.path.join(data_folder, 'auto_dataset_metadata.json')
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"\n📊 Dataset loaded:")
        print(f"  Total samples: {metadata['num_samples']}")
        print(f"  Counter types: {metadata['counter_patterns']}")
    else:
        print("❌ Metadata not found")
        return
    
    # Create simple model
    print("\nCreating model weights...")
    model_weights = create_simple_model()
    
    # Save model
    os.makedirs('models', exist_ok=True)
    
    model_file = 'models/simple_model.json'
    with open(model_file, 'w') as f:
        json.dump(model_weights, f, indent=2)
    
    print(f"\n✅ Simple model saved to {model_file}")
    print("\n📝 Note: This is a demo model.")
    print("For real AI training, install TensorFlow 2.15.0:")
    print("pip install tensorflow==2.15.0")
    
    # Show next steps
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Install TensorFlow: pip install tensorflow==2.15.0")
    print("2. Run: python scripts/auto_train_model_real.py")
    print("3. Test: streamlit run app_simple_fixed.py")

if __name__ == "__main__":
    main()