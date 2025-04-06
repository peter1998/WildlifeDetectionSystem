from flask import Blueprint, jsonify, current_app, redirect, url_for, render_template
from app.models.models import Image, Species, Annotation
from app import db
from sqlalchemy import func, distinct
import platform
import sys
import flask
import sqlalchemy
import os

# Create a blueprint for main routes
main = Blueprint('main', __name__)

# Add dashboard route to render the main page with accurate statistics
@main.route('/')
def dashboard():
    """Render the main dashboard with accurate statistics."""
    # Get statistics directly from database
    stats = get_system_statistics()
    
    # Pass data directly to template
    return render_template(
        'index.html',  # Your dashboard template
        total_images=stats['total_images'],
        annotated_images=stats['annotated_images'],
        species_count=stats['species_count'],
        folder_count=stats['folder_count'],
        annotation_percentage=stats['annotation_percentage']
    )

def get_system_statistics():
    """Get accurate system statistics directly from the database."""
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
    
    # Calculate annotation percentage for progress bar
    if stats['total_images'] > 0:
        stats['annotation_percentage'] = (stats['annotated_images'] / stats['total_images']) * 100
    else:
        stats['annotation_percentage'] = 0
    
    return stats

# Your existing API endpoints
@main.route('/api')
def index():
    """Root endpoint for the API."""
    return jsonify({
        'message': 'Welcome to the Wildlife Detection System API',
        'status': 'operational'
    })

@main.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy'
    })

@main.route('/system-info')
def system_info():
    """Provide system and application diagnostic information."""
    from app import db  # Import db from your app module

    return jsonify({
        'python_version': platform.python_version(),
        'flask_version': flask.__version__,
        'sqlalchemy_version': sqlalchemy.__version__,
        'debug_mode': current_app.debug if current_app else 'N/A',
        'config': str(current_app.config) if current_app else 'N/A',
        'database_uri': current_app.config.get('SQLALCHEMY_DATABASE_URI', 'N/A') if current_app else 'N/A'
    })