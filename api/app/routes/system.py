from flask import Blueprint, jsonify, current_app
from app.models.models import Image, Species, Annotation
from app import db
from sqlalchemy import func, distinct
import os

# Create a blueprint for system routes
system = Blueprint('system', __name__)

@system.route('/api/system/stats')
def get_system_stats():
    """Get comprehensive system statistics"""
    try:
        # Get statistics directly from database
        stats = {}
        
        # Total Images
        stats['total_images'] = Image.query.count()
        
        # Annotated Images (images with at least one annotation)
        stats['annotated_images'] = db.session.query(
            func.count(distinct(Annotation.image_id))
        ).scalar() or 0
        
        # Species Count
        stats['species_count'] = Species.query.count()
        
        # Image Folders Count
        upload_folder = None
        try:
            # Try to get the upload folder from app config
            upload_folder = current_app.config.get('UPLOAD_FOLDER')
        except:
            # Fallback to common locations if not found in config
            possible_locations = [
                'data/raw_images',
                'raw_images',
                'static/uploads',
                'uploads'
            ]
            
            for loc in possible_locations:
                if os.path.exists(loc) and os.path.isdir(loc):
                    upload_folder = loc
                    break
        
        # Count folders in the upload directory
        if upload_folder and os.path.exists(upload_folder):
            # Count only directories, not files
            stats['folder_count'] = sum(
                1 for item in os.listdir(upload_folder) 
                if os.path.isdir(os.path.join(upload_folder, item))
            )
        else:
            stats['folder_count'] = 0
        
        # Total annotations count
        stats['total_annotations'] = Annotation.query.count()
        
        # Calculate annotation percentage
        if stats['total_images'] > 0:
            stats['annotation_percentage'] = (stats['annotated_images'] / stats['total_images']) * 100
        else:
            stats['annotation_percentage'] = 0
            
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500