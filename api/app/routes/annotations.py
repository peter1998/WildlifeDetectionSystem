from flask import Blueprint, request, jsonify, current_app
import os
import json
from app import db
from app.models.models import Annotation, Image, Species
from app.services.annotation_service import AnnotationService

# Create blueprint for annotation routes
annotations = Blueprint('annotations', __name__, url_prefix='/api/annotations')

@annotations.route('/image/<int:image_id>', methods=['GET'])
def get_annotations_by_image(image_id):
    """Get all annotations for a specific image."""
    try:
        # Check if image exists
        image = Image.query.get_or_404(image_id)
        
        # Get annotations
        annotations = AnnotationService.get_annotations_by_image_id(image_id)
        
        # Format response
        result = {
            'success': True,
            'annotations': [{
                'id': ann.id,
                'image_id': ann.image_id,
                'species_id': ann.species_id,
                'x_min': ann.x_min,
                'y_min': ann.y_min,
                'x_max': ann.x_max,
                'y_max': ann.y_max,
                'confidence': ann.confidence,
                'is_verified': ann.is_verified,
                'created_at': ann.created_at.isoformat(),
                'updated_at': ann.updated_at.isoformat()
            } for ann in annotations]
        }
        
        return jsonify(result), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@annotations.route('/<int:annotation_id>', methods=['GET'])
def get_annotation(annotation_id):
    """Get a specific annotation by ID."""
    try:
        annotation = AnnotationService.get_annotation_by_id(annotation_id)
        
        # Format response
        result = {
            'success': True,
            'annotation': {
                'id': annotation.id,
                'image_id': annotation.image_id,
                'species_id': annotation.species_id,
                'x_min': annotation.x_min,
                'y_min': annotation.y_min,
                'x_max': annotation.x_max,
                'y_max': annotation.y_max,
                'confidence': annotation.confidence,
                'is_verified': annotation.is_verified,
                'created_at': annotation.created_at.isoformat(),
                'updated_at': annotation.updated_at.isoformat()
            }
        }
        
        return jsonify(result), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@annotations.route('/', methods=['POST'])
def create_annotation():
    """Create a new annotation."""
    data = request.json
    
    # Validate required fields
    required_fields = ['image_id', 'species_id', 'x_min', 'y_min', 'x_max', 'y_max']
    if not data or not all(field in data for field in required_fields):
        return jsonify({
            'success': False,
            'message': f'Missing required fields: {", ".join(required_fields)}'
        }), 400
    
    try:
        # Create annotation
        annotation = AnnotationService.create_annotation(
            image_id=data['image_id'],
            species_id=data['species_id'],
            x_min=data['x_min'],
            y_min=data['y_min'],
            x_max=data['x_max'],
            y_max=data['y_max'],
            confidence=data.get('confidence'),
            is_verified=data.get('is_verified', False)
        )
        
        # Format response
        result = {
            'success': True,
            'message': 'Annotation created successfully',
            'annotation': {
                'id': annotation.id,
                'image_id': annotation.image_id,
                'species_id': annotation.species_id,
                'x_min': annotation.x_min,
                'y_min': annotation.y_min,
                'x_max': annotation.x_max,
                'y_max': annotation.y_max,
                'confidence': annotation.confidence,
                'is_verified': annotation.is_verified,
                'created_at': annotation.created_at.isoformat(),
                'updated_at': annotation.updated_at.isoformat()
            }
        }
        
        return jsonify(result), 201
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@annotations.route('/batch', methods=['POST'])
def create_batch_annotations():
    """Create multiple annotations for an image at once."""
    data = request.json
    
    if not data or 'image_id' not in data or 'annotations' not in data:
        return jsonify({
            'success': False,
            'message': 'Missing required fields: image_id and annotations'
        }), 400
    
    try:
        # First validate the image exists
        image = Image.query.get_or_404(data['image_id'])
        
        # Create the annotations
        created_annotations = AnnotationService.create_batch_annotations(
            image_id=data['image_id'],
            annotations_data=data['annotations']
        )
        
        return jsonify({
            'success': True,
            'message': f'Created {len(created_annotations)} annotations',
            'annotations': [
                {
                    'id': ann.id,
                    'image_id': ann.image_id,
                    'species_id': ann.species_id,
                    'x_min': ann.x_min,
                    'y_min': ann.y_min,
                    'x_max': ann.x_max,
                    'y_max': ann.y_max,
                    'confidence': ann.confidence,
                    'is_verified': ann.is_verified
                } for ann in created_annotations
            ]
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@annotations.route('/<int:annotation_id>', methods=['PUT'])
def update_annotation(annotation_id):
    """Update an existing annotation."""
    data = request.json
    
    if not data:
        return jsonify({
            'success': False,
            'message': 'No data provided'
        }), 400
    
    try:
        # Update annotation
        annotation = AnnotationService.update_annotation(
            annotation_id=annotation_id,
            species_id=data.get('species_id'),
            x_min=data.get('x_min'),
            y_min=data.get('y_min'),
            x_max=data.get('x_max'),
            y_max=data.get('y_max'),
            confidence=data.get('confidence'),
            is_verified=data.get('is_verified')
        )
        
        # Format response
        result = {
            'success': True,
            'message': 'Annotation updated successfully',
            'annotation': {
                'id': annotation.id,
                'image_id': annotation.image_id,
                'species_id': annotation.species_id,
                'x_min': annotation.x_min,
                'y_min': annotation.y_min,
                'x_max': annotation.x_max,
                'y_max': annotation.y_max,
                'confidence': annotation.confidence,
                'is_verified': annotation.is_verified,
                'created_at': annotation.created_at.isoformat(),
                'updated_at': annotation.updated_at.isoformat()
            }
        }
        
        return jsonify(result), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@annotations.route('/<int:annotation_id>', methods=['DELETE'])
def delete_annotation(annotation_id):
    """Delete an annotation."""
    try:
        success = AnnotationService.delete_annotation(annotation_id)
        
        return jsonify({
            'success': success,
            'message': 'Annotation deleted successfully'
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@annotations.route('/export', methods=['GET'])
def export_annotations():
    """Export annotations in specified format."""
    format = request.args.get('format', 'coco')
    
    try:
        if format.lower() == 'coco':
            output_file = os.path.join(current_app.config['ANNOTATIONS_FOLDER'], 'coco_annotations.json')
            result = AnnotationService.export_coco_format(output_file)
            
            return jsonify({
                'success': True,
                'message': 'Annotations exported successfully in COCO format',
                'file_path': output_file
            }), 200
        else:
            return jsonify({
                'success': False,
                'message': f'Unsupported format: {format}'
            }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@annotations.route('/export/yolo', methods=['GET'])
def export_yolo():
    """Export annotations in YOLO format."""
    try:
        output_dir = request.args.get('output_dir', os.path.join(current_app.config['EXPORT_DIR'], 'yolo_export'))
        
        result = AnnotationService.export_yolo_format(output_dir)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500
