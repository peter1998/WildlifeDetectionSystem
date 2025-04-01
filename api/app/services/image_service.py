import os
import uuid
from datetime import datetime
from flask import current_app
from werkzeug.utils import secure_filename
from app import db
from app.models.models import Image
from app.utils.image_utils import get_image_dimensions, extract_image_metadata

class ImageService:
    """Service for managing camera trap images."""
    
    @staticmethod
    def allowed_file(filename):
        """Check if the file has an allowed extension."""
        ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    @staticmethod
    def save_uploaded_file(file, location=None, camera_id=None):
        """Save an uploaded file and create a database record for it."""
        if file and ImageService.allowed_file(file.filename):
            # Generate unique filename
            original_filename = secure_filename(file.filename)
            extension = original_filename.rsplit('.', 1)[1].lower()
            unique_filename = f"{str(uuid.uuid4())}.{extension}"
            
            # Save file to upload folder
            upload_folder = current_app.config['UPLOAD_FOLDER']
            os.makedirs(upload_folder, exist_ok=True)
            file_path = os.path.join(upload_folder, unique_filename)
            file.save(file_path)
            
            # Get image dimensions and metadata
            width, height = get_image_dimensions(file_path)
            metadata = extract_image_metadata(file_path)
            
            # Use provided values or defaults
            location = location or metadata.get('location')
            camera_id = camera_id or metadata.get('camera_id')
            timestamp = metadata.get('timestamp')
            
            # Create database record
            new_image = Image(
                filename=unique_filename,
                original_path=file_path,
                width=width,
                height=height,
                location=location,
                camera_id=camera_id,
                timestamp=timestamp
            )
            
            try:
                db.session.add(new_image)
                db.session.commit()
                return new_image
            except Exception as e:
                db.session.rollback()
                # Delete the file if database operation failed
                if os.path.exists(file_path):
                    os.remove(file_path)
                raise e
        
        return None
    
    @staticmethod
    def get_all_images(page=1, per_page=20, folder=''):
        """
        Get a paginated list of all images.
        
        Args:
            page (int): Page number (1-indexed)
            per_page (int): Number of items per page
            folder (str): Filter by folder
            
        Returns:
            tuple: (list of images, total count)
        """
        query = Image.query
        
        # Apply folder filter if provided
        if folder:
            query = query.filter(Image.filename.like(f"{folder}/%"))
        
        # Order by upload date (newest first)
        query = query.order_by(Image.upload_date.desc())
        
        # Paginate the results
        paginated = query.paginate(page=page, per_page=per_page, error_out=False)
        
        return paginated.items, paginated.total
    
    @staticmethod
    def get_image_by_id(image_id):
        """Get an image by its ID."""
        return Image.query.get(image_id)
