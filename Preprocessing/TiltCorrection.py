import sys
import os

# Ensure the parent directories are accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import cv2
import numpy as np
from scipy.stats import linregress

# Works!

def find_text_tilt(image_path):
    """Finds the tilt angle of text in an image by detecting the central text axis."""
    
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Apply binary threshold
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours (bounding boxes around characters)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Collect the center of each bounding box
    centers = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2  # Compute centroid
        centers.append((cx, cy))

    if len(centers) < 2:
        raise ValueError("Not enough characters detected to compute tilt.")

    # Convert to numpy array for processing
    centers = np.array(centers)

    # Perform linear regression (fit a line through the character centers)
    slope, intercept, _, _, _ = linregress(centers[:, 0], centers[:, 1])

    # Compute tilt angle (convert slope to degrees)
    angle = np.arctan(slope) * (180 / np.pi)

    return angle

def deskew_image(image_path, save_path=os.path.join(os.getcwd(), "cache/deskewed_image.png")):
    """Rotates an image to correct tilt using the detected tilt angle."""
    
    # Load the original image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Find tilt angle
    tilt_angle = find_text_tilt(image_path)

    # Get image dimensions
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    # Compute rotation matrix (negative angle to deskew)
    rotation_matrix = cv2.getRotationMatrix2D(center, tilt_angle, 1.0)

    # Perform affine transformation (rotation)
    deskewed_img = cv2.warpAffine(img, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Save and return the deskewed image
    cv2.imwrite(save_path, deskewed_img)
    # print(f"Deskewed image saved as {save_path}")

    return deskewed_img
