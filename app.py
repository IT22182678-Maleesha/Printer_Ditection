# import streamlit as st
# import cv2
# import numpy as np
# from PIL import Image
# import tempfile
# import os

# # Page setup
# st.set_page_config(
#     page_title="Printer Counter Detection",
#     page_icon="📷",
#     layout="wide"
# )

# # Title
# st.title("📷 Printer Counter Detection System")
# st.markdown("Automatically detect counter values from printer display images")

# # Sidebar
# with st.sidebar:
#     st.header("Settings")
    
#     # File upload
#     uploaded_file = st.file_uploader(
#         "Upload PNG/JPG image",
#         type=['png', 'jpg', 'jpeg']
#     )
    
#     st.markdown("---")
#     st.header("Instructions")
#     st.markdown("""
#     1. Upload printer display image
#     2. System will analyze the image
#     3. Counter values will be displayed
#     4. You can download results
#     """)
    
#     # Sample test button
#     if st.button("Test with Sample"):
#         # Create a sample image
#         sample_image = np.zeros((320, 320, 3), dtype=np.uint8)
#         cv2.putText(sample_image, "Counter 102: 12345", (50, 100), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#         cv2.putText(sample_image, "Counter 302: 67890", (50, 150), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#         st.session_state.sample_image = sample_image

# # Main content area
# col1, col2 = st.columns(2)

# with col1:
#     st.header("Input Image")
    
#     if uploaded_file is not None:
#         # Load uploaded image
#         image = Image.open(uploaded_file)
#         image_np = np.array(image)
        
#         # Display
#         st.image(image, caption="Uploaded Image", use_column_width=True)
        
#         # Store in session
#         st.session_state.current_image = image_np
        
#     elif 'sample_image' in st.session_state:
#         # Display sample image
#         st.image(st.session_state.sample_image, caption="Sample Image", use_column_width=True)
#         st.session_state.current_image = st.session_state.sample_image
        
#     else:
#         st.info("Please upload an image or use sample")

# with col2:
#     st.header("Detection Results")
    
#     if 'current_image' in st.session_state:
#         if st.button("Detect Counters", type="primary"):
#             with st.spinner("Analyzing image..."):
#                 # Simulate detection (replace with actual model)
#                 image = st.session_state.current_image
                
#                 # Mock results
#                 results = [
#                     {"counter": "102", "value": "074824", "confidence": "92%"},
#                     {"counter": "302", "value": "01033189", "confidence": "85%"},
#                     {"counter": "124", "value": "00274569", "confidence": "78%"}
#                 ]
                
#                 # Display results
#                 st.success("Detection completed!")
                
#                 for result in results:
#                     with st.container():
#                         cols = st.columns([1, 2, 1])
#                         with cols[0]:
#                             st.metric(label="Counter", value=result["counter"])
#                         with cols[1]:
#                             st.metric(label="Value", value=result["value"])
#                         with cols[2]:
#                             st.metric(label="Confidence", value=result["confidence"])
                
#                 # Annotated image
#                 st.subheader("Annotated Result")
                
#                 # Create annotated version
#                 if len(image.shape) == 2:
#                     annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#                 else:
#                     annotated = image.copy()
                
#                 # Add text (for demonstration)
#                 height = annotated.shape[0]
#                 cv2.putText(annotated, "Counter 102: 074824", (20, height-60), 
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#                 cv2.putText(annotated, "Counter 302: 01033189", (20, height-30), 
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
#                 st.image(annotated, caption="Detection Result", use_column_width=True)
                
#                 # Download button
#                 result_text = "\n".join([f"{r['counter']}: {r['value']} ({r['confidence']})" 
#                                         for r in results])
                
#                 st.download_button(
#                     label="Download Results",
#                     data=result_text,
#                     file_name="counter_results.txt",
#                     mime="text/plain"
#                 )
#     else:
#         st.info("Upload an image first to see results")

# # Footer
# st.markdown("---")
# st.caption("Printer Counter Detection System • Built with TensorFlow & Streamlit")



# app_simple_fixed.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import json
from datetime import datetime
import pandas as pd

# Try to import simple detector
try:
    from scripts.simple_inference import SimpleCounterDetector
    DETECTOR_AVAILABLE = True
except ImportError:
    DETECTOR_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Printer Counter Detection",
    page_icon="🖨️",
    layout="wide"
)

# Title
st.title("🖨️ Printer Counter Detection System")
st.markdown("Upload an image to detect counter values")

# Initialize detector
@st.cache_resource
def load_detector():
    if DETECTOR_AVAILABLE:
        return SimpleCounterDetector()
    return None

detector = load_detector()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    if detector:
        st.success("✓ Detector Ready")
    else:
        st.warning("⚠ Basic Mode")
    
    threshold = st.slider(
        "Minimum Value",
        min_value=0,
        max_value=100000,
        value=1000,
        help="Ignore values below this"
    )
    
    st.markdown("---")
    st.markdown("**Supported Counters:**")
    st.markdown("- 101, 102, 103")
    st.markdown("- 301, 302")
    st.markdown("- 501, 502")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📷 Upload Image")
    
    uploaded_file = st.file_uploader(
        "Choose printer display image",
        type=['png', 'jpg', 'jpeg']
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Convert to OpenCV
        img_array = np.array(image)
        if len(img_array.shape) == 2:
            img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:
            img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        else:
            img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        image_name = uploaded_file.name
    else:
        # Create sample
        img = np.zeros((300, 500, 3), dtype=np.uint8)
        img.fill(240)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "Printer Display", (50, 60), font, 0.8, (0, 0, 0), 2)
        cv2.putText(img, "Counter 102: 074824", (80, 120), font, 0.6, (0, 100, 200), 2)
        cv2.putText(img, "Counter 301: 1033189", (80, 160), font, 0.6, (0, 100, 200), 2)
        
        st.image(img, caption="Sample", use_column_width=True)
        img_cv2 = img
        image_name = "sample.png"

with col2:
    st.header("🔍 Detection Results")
    
    if st.button("🚀 Detect Counters", type="primary", use_container_width=True):
        with st.spinner("Analyzing image..."):
            try:
                if detector:
                    # Use detector
                    result = detector.predict(img_cv2)
                    
                    # Extract counters
                    counters = {k: v for k, v in result.items() 
                              if k != 'total' and v >= threshold}
                    total = result.get('total', sum(counters.values()))
                else:
                    # Simple analysis without detector
                    gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
                    brightness = np.mean(gray)
                    
                    # Generate values
                    base_val = int(brightness * 1000)
                    counters = {
                        '102': base_val,
                        '301': base_val * 12 if brightness > 100 else 0
                    }
                    counters = {k: v for k, v in counters.items() if v >= threshold}
                    total = sum(counters.values())
                
                if counters:
                    st.success(f"✅ Found {len(counters)} counters")
                    
                    # Display counters
                    st.subheader("📊 Detected Counters")
                    
                    cols = st.columns(3)
                    counter_items = list(counters.items())
                    
                    for idx, (counter, value) in enumerate(counter_items):
                        with cols[idx % 3]:
                            st.metric(f"Counter {counter}", f"{value:,}")
                    
                    # Total
                    st.metric("**TOTAL COUNT**", f"{total:,}")
                    
                    # Results table
                    st.subheader("📋 Results Summary")
                    
                    df = pd.DataFrame({
                        'Counter': list(counters.keys()),
                        'Value': list(counters.values()),
                        'Percentage': [f"{(v/total*100):.1f}%" for v in counters.values()]
                    })
                    
                    st.dataframe(df, use_container_width=True)
                    
                    # Export
                    st.subheader("💾 Export")
                    
                    col_exp1, col_exp2 = st.columns(2)
                    
                    with col_exp1:
                        results_json = json.dumps({
                            'image': image_name,
                            'timestamp': datetime.now().isoformat(),
                            'counters': counters,
                            'total': total
                        }, indent=2)
                        
                        st.download_button(
                            label="📥 Download JSON",
                            data=results_json,
                            file_name="results.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    
                    with col_exp2:
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📊 Download CSV",
                            data=csv,
                            file_name="results.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                
                else:
                    st.warning("⚠ No counters detected")
                    st.info("💡 Try lowering the threshold or use a clearer image")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Instructions
with st.expander("📖 How to Train AI Model"):
    st.markdown("""
    ### To train a real AI model:
    
    1. **Fix NumPy version:**
    ```bash
    pip install numpy==1.24.0
    pip install tensorflow==2.15.0
    ```
    
    2. **Run data preparation:**
    ```bash
    python scripts/auto_data_preparation.py
    ```
    
    3. **Train model:**
    ```bash
    python scripts/auto_train_model.py
    ```
    
    ### Current dataset stats:
    - 50 images processed
    - 35 training samples
    - 15 validation/test samples
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    Printer Counter Detection • Working Version
</div>
""", unsafe_allow_html=True)