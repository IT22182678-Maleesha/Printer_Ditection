# """
# Train a model using synthetic data based on your actual counter patterns
# """

# import tensorflow as tf
# import numpy as np
# import os
# import json
# import cv2
# from sklearn.model_selection import train_test_split
# from datetime import datetime

# print("="*60)
# print("TRAINING PRINTER COUNTER MODEL")
# print("="*60)

# def create_synthetic_dataset(num_samples=1000):
#     """Create synthetic training data based on printer display patterns"""
    
#     images = []
#     labels = []
    
#     # Counter types we want to detect
#     counter_types = ['101', '102', '103', '112', '113', '114', '118', 
#                     '201', '203', '301', '302', '401', '402', '501', '502']
    
#     print(f"\nCreating {num_samples} synthetic training samples...")
    
#     for i in range(num_samples):
#         # Create a printer-like display
#         img = np.zeros((224, 224, 3), dtype=np.float32)
        
#         # Background (light gray)
#         img.fill(0.9)
        
#         # Add some random rectangles (simulating display segments)
#         for _ in range(np.random.randint(3, 8)):
#             x = np.random.randint(20, 180)
#             y = np.random.randint(20, 180)
#             w = np.random.randint(40, 80)
#             h = np.random.randint(20, 40)
#             color = np.random.uniform(0.7, 0.9, 3)
#             cv2.rectangle(img, (x, y), (x+w, y+h), color, -1)
        
#         # Add noise (simulating image capture artifacts)
#         noise = np.random.normal(0, 0.05, img.shape)
#         img = np.clip(img + noise, 0, 1)
        
#         # Create labels - random counter values
#         counter_values = []
        
#         for counter in counter_types:
#             # Some counters appear more frequently
#             if counter in ['101', '102', '301', '501']:
#                 if np.random.random() > 0.3:  # 70% chance
#                     value = np.random.randint(1000, 999999)
#                 else:
#                     value = 0
#             else:
#                 if np.random.random() > 0.7:  # 30% chance
#                     value = np.random.randint(1000, 99999)
#                 else:
#                     value = 0
            
#             counter_values.append(value)
        
#         images.append(img)
#         labels.append(counter_values)
        
#         if (i + 1) % 100 == 0:
#             print(f"  Created {i + 1}/{num_samples} samples")
    
#     return np.array(images), np.array(labels), counter_types

# def build_model(num_counters=15):
#     """Build a CNN model for counter value prediction"""
    
#     inputs = tf.keras.Input(shape=(224, 224, 3))
    
#     # Feature extraction
#     x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.MaxPooling2D((2, 2))(x)
#     x = tf.keras.layers.Dropout(0.2)(x)
    
#     x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.MaxPooling2D((2, 2))(x)
#     x = tf.keras.layers.Dropout(0.3)(x)
    
#     x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.GlobalAveragePooling2D()(x)
#     x = tf.keras.layers.Dropout(0.4)(x)
    
#     # Multiple outputs for different counters
#     outputs = []
#     counter_names = ['101', '102', '103', '112', '113', '114', '118', 
#                     '201', '203', '301', '302', '401', '402', '501', '502']
    
#     # Shared dense layers
#     x_dense = tf.keras.layers.Dense(256, activation='relu')(x)
#     x_dense = tf.keras.layers.Dropout(0.3)(x_dense)
#     x_dense = tf.keras.layers.Dense(128, activation='relu')(x_dense)
    
#     for i in range(num_counters):
#         counter_output = tf.keras.layers.Dense(64, activation='relu')(x_dense)
#         counter_output = tf.keras.layers.Dense(1, activation='linear', 
#                                               name=f'counter_{counter_names[i]}')(counter_output)
#         outputs.append(counter_output)
    
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
#     return model

# def main():
#     # Check GPU
#     print(f"\n1. System check:")
#     print(f"   TensorFlow version: {tf.__version__}")
#     gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
#     print(f"   GPU available: {gpu_available}")
    
#     # Create synthetic dataset
#     print("\n2. Creating dataset...")
#     images, labels, counter_types = create_synthetic_dataset(num_samples=2000)
    
#     # Split data
#     print("\n3. Splitting data...")
#     X_train, X_temp, y_train, y_temp = train_test_split(
#         images, labels, test_size=0.3, random_state=42
#     )
#     X_val, X_test, y_val, y_test = train_test_split(
#         X_temp, y_temp, test_size=0.5, random_state=42
#     )
    
#     print(f"   Training samples: {X_train.shape[0]}")
#     print(f"   Validation samples: {X_val.shape[0]}")
#     print(f"   Test samples: {X_test.shape[0]}")
    
#     # Build model
#     print("\n4. Building model...")
#     model = build_model(num_counters=len(counter_types))
    
#     # Compile model
#     print("\n5. Compiling model...")
    
#     # Create loss and metric dictionaries
#     losses = {}
#     metrics = {}
    
#     for i, counter in enumerate(counter_types):
#         losses[f'counter_{counter}'] = 'mse'
#         metrics[f'counter_{counter}'] = 'mae'
    
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#         loss=losses,
#         metrics=metrics
#     )
    
#     model.summary()
    
#     # Create callbacks
#     print("\n6. Setting up training...")
#     callbacks = [
#         tf.keras.callbacks.EarlyStopping(
#             monitor='val_loss',
#             patience=10,
#             restore_best_weights=True
#         ),
#         tf.keras.callbacks.ReduceLROnPlateau(
#             monitor='val_loss',
#             factor=0.5,
#             patience=5,
#             min_lr=1e-6
#         ),
#         tf.keras.callbacks.ModelCheckpoint(
#             filepath='models/checkpoints/best_model_epoch_{epoch:02d}.h5',
#             monitor='val_loss',
#             save_best_only=True
#         ),
#         tf.keras.callbacks.TensorBoard(
#             log_dir=f'logs/{datetime.now().strftime("%Y%m%d-%H%M%S")}',
#             histogram_freq=1
#         )
#     ]
    
#     # Prepare output dictionaries for training
#     train_outputs = {}
#     val_outputs = {}
    
#     for i, counter in enumerate(counter_types):
#         train_outputs[f'counter_{counter}'] = y_train[:, i]
#         val_outputs[f'counter_{counter}'] = y_val[:, i]
    
#     # Train model
#     print("\n7. Training model...")
#     history = model.fit(
#         X_train,
#         train_outputs,
#         validation_data=(X_val, val_outputs),
#         epochs=50,
#         batch_size=32,
#         callbacks=callbacks,
#         verbose=1
#     )
    
#     # Save model
#     print("\n8. Saving model...")
#     os.makedirs('models/trained', exist_ok=True)
    
#     # Save as H5
#     model.save('models/trained/printer_counter_model.h5')
    
#     # Save as SavedModel
#     model.save('models/trained/saved_model', save_format='tf')
    
#     # Convert to TFLite
#     print("\n9. Converting to TFLite...")
#     converter = tf.lite.TFLiteConverter.from_keras_model(model)
#     converter.optimizations = [tf.lite.Optimize.DEFAULT]
#     tflite_model = converter.convert()
    
#     with open('models/trained/printer_counter_model.tflite', 'wb') as f:
#         f.write(tflite_model)
    
#     print("   ✓ Model saved: models/trained/printer_counter_model.h5")
#     print("   ✓ TFLite model saved: models/trained/printer_counter_model.tflite")
    
#     # Test the model
#     print("\n10. Testing model...")
#     test_loss = model.evaluate(X_test, {
#         f'counter_{counter}': y_test[:, i] for i, counter in enumerate(counter_types)
#     }, verbose=0)
    
#     print(f"   Test Loss: {test_loss[0]:.4f}")
    
#     # Make predictions on a few samples
#     print("\n11. Sample predictions:")
#     sample_indices = np.random.choice(len(X_test), 3, replace=False)
    
#     for idx in sample_indices:
#         sample = X_test[idx:idx+1]
#         predictions = model.predict(sample, verbose=0)
        
#         print(f"\n   Sample {idx + 1}:")
#         for i, counter in enumerate(counter_types[:5]):  # Show first 5
#             pred_value = predictions[i][0][0]
#             true_value = y_test[idx, i]
#             if true_value > 0 or pred_value > 100:  # Show if relevant
#                 print(f"     Counter {counter}: Predicted {pred_value:.0f}, Actual {true_value}")
    
#     # Save training history
#     history_dict = {
#         'loss': [float(v) for v in history.history['loss']],
#         'val_loss': [float(v) for v in history.history['val_loss']]
#     }
    
#     with open('models/training_history.json', 'w') as f:
#         json.dump(history_dict, f, indent=4)
    
#     print("\n" + "="*60)
#     print("TRAINING COMPLETED SUCCESSFULLY!")
#     print("="*60)
    
#     print("\nNext steps:")
#     print("1. Run: streamlit run app.py")
#     print("2. Test with your actual images")
#     print("3. Fine-tune with real data when available")

# if __name__ == "__main__":
#     main()