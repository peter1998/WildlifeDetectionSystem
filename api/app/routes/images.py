from flask import Blueprint, request, jsonify, current_app, send_file
import os
from app.services.image_service import ImageService
from app.models.models import Image, Annotation
from app import db

# Create blueprint for image routes
images = Blueprint('images', __name__, url_prefix='/api/images')

@images.route('/<int:image_id>/file', methods=['GET'])
def get_image_file(image_id):
    """Get the image file."""
    try:
        # Get the image
        image = Image.query.get_or_404(image_id)
        
        # Check if file exists
        if not os.path.exists(image.original_path):
            return jsonify({
                'success': False,
                'message': 'Image file not found'
            }), 404
        
        # Get the file extension to determine content type
        _, ext = os.path.splitext(image.original_path)
        ext = ext.lower()
        
        content_type = 'image/jpeg'  # Default
        if ext == '.png':
            content_type = 'image/png'
        elif ext == '.gif':
            content_type = 'image/gif'
        elif ext == '.tif' or ext == '.tiff':
            content_type = 'image/tiff'
        
        # Return the file
        return send_file(
            image.original_path, 
            mimetype=content_type
        )
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@images.route('/<int:image_id>/no-animals', methods=['POST'])
def mark_no_animals(image_id):
    """Mark an image as having no animals."""
    try:
        # Get the image
        image = Image.query.get_or_404(image_id)
        
        # Delete any existing annotations
        Annotation.query.filter_by(image_id=image_id).delete()
        
        # You might want to add a field to your Image model for this
        # For example: image.has_animals = False
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Image {image_id} marked as having no animals'
        })
    
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@images.route('/folders', methods=['GET'])
def get_folders():
    """Get list of folders containing images."""
    folders = set()
    
    for image in Image.query.all():
        if image.filename and '/' in image.filename:
            folder = image.filename.split('/')[0]
            folders.add(folder)
    
    return jsonify({
        'success': True,
        'folders': sorted(list(folders))
    })


@images.route('/', methods=['POST'])
def upload_image():
    """Upload a new camera trap image."""
    # Check if the post request has a file part
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'message': 'No file part in the request'
        }), 400
        
    file = request.files['file']
    
    # If user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return jsonify({
            'success': False,
            'message': 'No selected file'
        }), 400
    
    # Get optional metadata
    location = request.form.get('location')
    camera_id = request.form.get('camera_id')
    
    try:
        image = ImageService.save_uploaded_file(file, location, camera_id)
        
        if image:
            return jsonify({
                'success': True,
                'message': 'File successfully uploaded',
                'image': {
                    'id': image.id,
                    'filename': image.filename,
                    'upload_date': image.upload_date.isoformat(),
                    'width': image.width,
                    'height': image.height
                }
            }), 201
        else:
            return jsonify({
                'success': False,
                'message': 'File type not allowed'
            }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error uploading file: {str(e)}'
        }), 500

@images.route('/', methods=['GET'])
def get_images():
    """Get a list of all uploaded images."""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    folder = request.args.get('folder', '')
    
    try:
        images, total = ImageService.get_all_images(page, per_page, folder)
        
        result = {
            'success': True,
            'total': total,
            'page': page,
            'per_page': per_page,
            'images': [{
                'id': img.id,
                'filename': img.filename,
                'upload_date': img.upload_date.isoformat(),
                'width': img.width,
                'height': img.height,
                'location': img.location,
                'camera_id': img.camera_id,
                'timestamp': img.timestamp.isoformat() if img.timestamp else None,
                'is_annotated': len(img.annotations) > 0  # Add this field to track annotated images
            } for img in images]
        }
        
        return jsonify(result), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error retrieving images: {str(e)}'
        }), 500

@images.route('/<int:image_id>', methods=['GET'])
def get_image(image_id):
    """Get a specific image by ID."""
    image = ImageService.get_image_by_id(image_id)
    
    if not image:
        return jsonify({
            'success': False,
            'message': 'Image not found'
        }), 404
    
    # Return image metadata, not the file
    return jsonify({
        'success': True,
        'image': {
            'id': image.id,
            'filename': image.filename,
            'upload_date': image.upload_date.isoformat(),
            'width': image.width,
            'height': image.height,
            'location': image.location,
            'camera_id': image.camera_id,
            'timestamp': image.timestamp.isoformat() if image.timestamp else None
        }
    })

@images.route('/index-existing', methods=['POST'])
def index_existing_images():
    """Index existing images in the raw_images directory."""
    from app.services.image_indexer import index_existing_images
    
    try:
        indexed, skipped = index_existing_images()
        
        return jsonify({
            'success': True,
            'message': f'Indexed {indexed} new images, skipped {skipped} existing images',
            'indexed_count': indexed,
            'skipped_count': skipped
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error indexing images: {str(e)}'
        }), 500
