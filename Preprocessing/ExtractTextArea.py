import sys
import os

# Ensure the parent directories are accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import cv2
import numpy as np
import os

def crop_to_text(image_path, save_path=os.path.join(os.getcwd(), "cache/cropped_image.png"), padding=10):
    """Crops the image to the region where text is written, removing extra space."""
    
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Apply binary threshold (invert if needed)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours of text
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find bounding box that contains all text
    x, y, w, h = cv2.boundingRect(np.vstack(contours))

    # Add padding (ensure within image bounds)
    x = max(x - padding, 0)
    y = max(y - padding, 0)
    w = min(w + 2 * padding, img.shape[1] - x)
    h = min(h + 2 * padding, img.shape[0] - y)

    # Crop the image
    cropped_img = img[y:y+h, x:x+w]

    # Save and return cropped image
    cv2.imwrite(save_path, cropped_img)
    # print(f"Cropped image saved as {save_path}")
    return cropped_img