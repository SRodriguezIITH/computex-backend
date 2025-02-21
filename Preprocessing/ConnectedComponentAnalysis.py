import sys
import os

# Ensure the parent directories are accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


import cv2
import numpy as np
import os

def connected_component_analysis(image_path, save_dir=os.path.join(os.getcwd(), "cache/final_letters"), output_size=(32, 32)):
    """Applies Connected Component Analysis (CCA) to segment characters in an image from left to right.
       Ensures all output images have the same dimensions.
    """

    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Apply binary threshold with inversion
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Perform Connected Component Analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # Create output directory if not exists
    os.makedirs(save_dir, exist_ok=True)

    # Create a color output image for visualization
    output = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # **Sort bounding boxes by x-coordinate (left to right)**
    sorted_indices = sorted(range(1, num_labels), key=lambda i: stats[i][0])  # Sorting by 'x' value

    # Loop through detected components in sorted order
    for i, idx in enumerate(sorted_indices):
        x, y, w, h, area = stats[idx]

        # Ignore very small components (likely noise)
        if area < 10:
            continue

        # Draw bounding box around each character
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract the individual character image
        char_img = binary[y:y+h, x:x+w]

        # **Resize while maintaining aspect ratio**
        char_img = resize_with_padding(char_img, output_size)

        # Save the resized character image
        char_filename = f"{save_dir}/char_{i+1}.png"
        cv2.imwrite(char_filename, char_img)

    # Save the labeled output image
    cv2.imwrite(os.path.join(os.getcwd(), "cache/cca_output.png"), output)
    # print(f"Identified {num_labels - 1} characters (sorted left to right).")
    # print(f'Final images have been saved to {save_dir}')

def resize_with_padding(image, target_size):
    """Resizes an image to the target size while maintaining aspect ratio by adding padding."""
    h, w = image.shape
    target_w, target_h = target_size

    # Calculate scale and new size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a blank canvas and place the resized image in the center
    padded = np.ones((target_h, target_w), dtype=np.uint8) * 0  # Black background
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return padded