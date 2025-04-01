import os
import cv2
import numpy as np
from PIL import Image, ExifTags
from datetime import datetime

def read_image(file_path):
    """
    Read an image file and return it as a numpy array.
    """
    return cv2.imread(file_path)

def save_image(image, output_path):
    """
    Save a numpy array as an image file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)

def get_image_dimensions(file_path):
    """
    Get the dimensions of an image.
    """
    img = cv2.imread(file_path)
    if img is None:
        # Attempt to use PIL if OpenCV fails
        try:
            with Image.open(file_path) as img_pil:
                return img_pil.size
        except Exception:
            return (None, None)
    
    height, width = img.shape[:2]
    return (width, height)

def extract_image_metadata(file_path):
    """
    Extract metadata from image EXIF tags.
    """
    metadata = {
        'timestamp': None,
        'camera_id': None,
        'location': None
    }
    
    try:
        with Image.open(file_path) as img:
            if hasattr(img, '_getexif') and img._getexif():
                exif = {
                    ExifTags.TAGS[k]: v
                    for k, v in img._getexif().items()
                    if k in ExifTags.TAGS
                }
                
                # Extract timestamp
                if 'DateTimeOriginal' in exif:
                    try:
                        metadata['timestamp'] = datetime.strptime(
                            exif['DateTimeOriginal'],
                            '%Y:%m:%d %H:%M:%S'
                        )
                    except ValueError:
                        pass
    except Exception as e:
        print(f"Error extracting metadata: {str(e)}")
    
    return metadata
