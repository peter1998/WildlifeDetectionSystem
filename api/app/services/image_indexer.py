import os
from app import db
from app.models.models import Image
from app.utils.image_utils import get_image_dimensions, extract_image_metadata
from flask import current_app

def index_existing_images():
    """
    Recursively scan the raw_images directory and add any images not already in the database.
    
    Returns:
        tuple: (number of images indexed, number of images skipped)
    """
    raw_images_dir = current_app.config['UPLOAD_FOLDER']
    
    # Get existing filenames from database
    existing_filenames = {img.filename for img in Image.query.all()}
    
    indexed_count = 0
    skipped_count = 0
    
    # Image extensions to look for
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', 
                       '.JPG', '.JPEG', '.PNG', '.TIF', '.TIFF']
    
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(raw_images_dir):
        for file in files:
            # Check if file has image extension
            if any(file.endswith(ext) for ext in image_extensions):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, raw_images_dir)
                
                # Skip if already in database (using relative path as unique identifier)
                if relative_path in existing_filenames:
                    skipped_count += 1
                    continue
                
                # Get image dimensions and metadata
                try:
                    width, height = get_image_dimensions(file_path)
                    metadata = extract_image_metadata(file_path)
                    
                    # Create database record
                    new_image = Image(
                        filename=relative_path,  # Store relative path as filename
                        original_path=file_path,
                        width=width,
                        height=height,
                        location=metadata.get('location'),
                        camera_id=metadata.get('camera_id'),
                        timestamp=metadata.get('timestamp')
                    )
                    
                    db.session.add(new_image)
                    indexed_count += 1
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    skipped_count += 1
    
    # Commit all changes at once
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"Database error: {str(e)}")
    
    return indexed_count, skipped_count
