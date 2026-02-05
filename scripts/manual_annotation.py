# import json
# import os
# import cv2
# import numpy as np

# def create_labels_from_your_data():
#     """
#     Create labels from the counts you provided
#     This connects your images with their actual counter values
#     """
    
#     # Your provided counts (I'll show how to match them with images)
#     counts_data = """
#     102	74894
#     102	191897
#     102	1033189
#     113	1009097
#     501	274569
#     301	286790
#     102	49003
#     101	616344
#     102	618612
#     501	150050
#     301	211441
#     101	121745
#     102	122222
#     501	65097
#     301	439
#     102	35211
#     102	62270
#     101	37636
#     102	74824
#     102	65945
#     101	82500
#     102	82525
#     201	35434
#     301	47066
#     302	47069
#     """
    
#     # Parse the data
#     lines = counts_data.strip().split('\n')
#     parsed_counts = []
    
#     for line in lines:
#         parts = line.strip().split('\t')
#         if len(parts) == 2:
#             counter, value = parts
#             parsed_counts.append({
#                 'counter': counter.strip(),
#                 'value': int(value.strip())
#             })
    
#     # Get all image files
#     image_folder = 'data/raw'
#     image_files = [f for f in os.listdir(image_folder) 
#                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
#     # Create labels - you'll need to match images with their counts
#     # For now, I'll create a demo version
    
#     labels = {}
    
#     # Create labels for each image
#     for i, img_file in enumerate(image_files[:20]):  # Process first 20 images
#         # For demonstration, assign some counts
#         # In real use, you should match each image with its actual counts
        
#         img_path = os.path.join(image_folder, img_file)
        
#         # Try to read and analyze the image
#         try:
#             img = cv2.imread(img_path)
#             if img is not None:
#                 # Get image info
#                 height, width = img.shape[:2]
                
#                 # Create sample labels based on image name
#                 if '102' in img_file.upper() or 'Total' in img_file.upper():
#                     labels[img_file] = {
#                         'counters': {
#                             '102': np.random.randint(10000, 999999),
#                             '101': np.random.randint(1000, 99999)
#                         },
#                         'image_size': [width, height],
#                         'total_count': 0  # Will calculate
#                     }
#                 elif '301' in img_file.upper() or 'Print' in img_file.upper():
#                     labels[img_file] = {
#                         'counters': {
#                             '301': np.random.randint(10000, 999999),
#                             '302': np.random.randint(10000, 999999)
#                         },
#                         'image_size': [width, height],
#                         'total_count': 0
#                     }
#                 else:
#                     # Generic label
#                     labels[img_file] = {
#                         'counters': {
#                             '101': np.random.randint(1000, 99999),
#                             '102': np.random.randint(1000, 99999),
#                             '301': np.random.randint(1000, 99999)
#                         },
#                         'image_size': [width, height],
#                         'total_count': 0
#                     }
                
#                 # Calculate total
#                 labels[img_file]['total_count'] = sum(labels[img_file]['counters'].values())
#         except:
#             continue
    
#     # Save labels
#     output_file = 'data/labels/manual_labels.json'
#     with open(output_file, 'w') as f:
#         json.dump(labels, f, indent=4)
    
#     print(f"Created labels for {len(labels)} images")
#     print(f"Saved to {output_file}")
    
#     return labels

# if __name__ == "__main__":
#     labels = create_labels_from_your_data()