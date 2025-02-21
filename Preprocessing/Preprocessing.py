import sys
import os

# Ensure the parent directories are accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from PIL import Image
from Preprocessing.TiltCorrection import deskew_image
from Preprocessing.ExtractTextArea import crop_to_text
from Preprocessing.ConnectedComponentAnalysis import connected_component_analysis

# from TiltCorrection import deskew_image
# from ExtractTextArea import crop_to_text
# from ConnectedComponentAnalysis import connected_component_analysis

CACHE_DIR = "cache"
FINAL_LETTERS_DIR = os.path.join(CACHE_DIR, "final_letters")

def ensure_directories_exist():
    """Create necessary directories if they do not exist."""
    for directory in [CACHE_DIR, FINAL_LETTERS_DIR]:
        os.makedirs(directory, exist_ok=True)

def preprocess_image(original_image):
    """Applies a sequence of preprocessing steps on an image."""
    ensure_directories_exist()
    
    deskewed_image = os.path.join(CACHE_DIR, "deskewed_image.png")
    cropped_image = os.path.join(CACHE_DIR, "cropped_image.png")

    try:
        deskew_image(original_image)
    except Exception as e:
        print(f"Error in tilt correction: {e}")
        return

    try:
        crop_to_text(deskewed_image)
    except Exception as e:
        print(f"Error in cropping text region: {e}")
        return

    try:
        connected_component_analysis(cropped_image)
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def save_to_cache(image_path, cache_dir=CACHE_DIR):
    """Saves a copy of the image to the cache directory."""
    ensure_directories_exist()
    
    cache_path = os.path.join(cache_dir, "image.png")
    try:
        image = Image.open(image_path)
        image.save(cache_path)
        return cache_path
    except Exception as e:
        print(f"Error saving image to cache: {e}")
        return None


def preprocess_image_from_input(image_path):
    image_path = image_path.strip()
    cached_image_path = save_to_cache(image_path)

    if cached_image_path:
        preprocess_image(cached_image_path)
    else:
        print("Failed to process image due to cache saving error.")


# print("Enter image path: ", end="")
# file = str(input())
# preprocess_image_from_input(file)