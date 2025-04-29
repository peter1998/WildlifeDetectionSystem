from flask import Blueprint, request, jsonify, current_app
import os
import json
from app import db
from app.models.models import Annotation, Image, Species
from app.services.annotation_service import AnnotationService
from datetime import datetime

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
    """Export annotations in COCO format."""
    try:
        # Generate a timestamped directory name if not specified
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = request.args.get('dataset', 'default')
        
        output_dir = request.args.get(
            'output_dir', 
            os.path.join(current_app.config['EXPORT_DIR'], f'coco_{dataset_name}_{timestamp}')
        )
        
        result = AnnotationService.export_coco_format(output_dir)
        
        return jsonify({
            'success': True,
            'message': 'Annotations exported successfully in COCO format',
            'output_dir': output_dir,
            'file_path': os.path.join(output_dir, 'annotations.json'),
            'stats': {
                'images': len(result.get('images', [])),
                'annotations': len(result.get('annotations', [])),
                'categories': len(result.get('categories', []))
            }
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@annotations.route('/export/yolo', methods=['GET'])
def export_yolo():
    """Export annotations in YOLO format."""
    try:
        # Generate a timestamped directory name if not specified
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = request.args.get('dataset', 'default')
        
        output_dir = request.args.get(
            'output_dir', 
            os.path.join(current_app.config['EXPORT_DIR'], f'yolo_{dataset_name}_{timestamp}')
        )
        
        result = AnnotationService.export_yolo_format(output_dir)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@annotations.route('/datasets', methods=['GET'])
def get_datasets():
    """Get list of annotated datasets (folders with images)."""
    try:
        datasets = AnnotationService.get_annotated_datasets()
        return jsonify({
            'success': True,
            'datasets': datasets
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@annotations.route('/exports', methods=['GET'])
def get_exports():
    """Get list of existing exports."""
    try:
        export_dir = current_app.config['EXPORT_DIR']
        exports = []
        
        if os.path.exists(export_dir):
            for item in os.listdir(export_dir):
                item_path = os.path.join(export_dir, item)
                if os.path.isdir(item_path):
                    # Identify export type from directory name or structure
                    export_type = "unknown"
                    if item.startswith("yolo_"):
                        export_type = "YOLO"
                    elif item.startswith("coco_"):
                        export_type = "COCO"
                    
                    # Get creation time
                    created = datetime.fromtimestamp(os.path.getctime(item_path)).isoformat()
                    
                    # Get size
                    total_size = 0
                    for dirpath, dirnames, filenames in os.walk(item_path):
                        for f in filenames:
                            fp = os.path.join(dirpath, f)
                            total_size += os.path.getsize(fp)
                    
                    exports.append({
                        'name': item,
                        'path': item_path,
                        'type': export_type,
                        'created': created,
                        'size_bytes': total_size
                    })
        
        return jsonify({
            'success': True,
            'exports': exports
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@annotations.route('/export/both', methods=['GET'])
def export_both_formats():
    """Export annotations in both YOLO and COCO formats in one call."""
    try:
        # Generate a timestamped directory name if not specified
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = request.args.get('dataset', 'default')
        
        # Export in YOLO format
        yolo_output_dir = os.path.join(current_app.config['EXPORT_DIR'], f'yolo_{dataset_name}_{timestamp}')
        yolo_result = AnnotationService.export_yolo_format(yolo_output_dir)
        
        # Export in COCO format
        coco_output_dir = os.path.join(current_app.config['EXPORT_DIR'], f'coco_{dataset_name}_{timestamp}')
        coco_result = AnnotationService.export_coco_format(coco_output_dir)
        
        return jsonify({
            'success': True,
            'message': 'Annotations exported successfully in both YOLO and COCO formats',
            'yolo_export': {
                'output_dir': yolo_output_dir,
                'stats': {
                    'images_count': yolo_result.get('images_count', 0),
                    'classes_count': yolo_result.get('classes_count', 0),
                    'train_images': yolo_result.get('train_images', 0),
                    'val_images': yolo_result.get('val_images', 0)
                }
            },
            'coco_export': {
                'output_dir': coco_output_dir,
                'file_path': os.path.join(coco_output_dir, 'annotations.json'),
                'stats': {
                    'images': len(coco_result.get('images', [])),
                    'annotations': len(coco_result.get('annotations', [])),
                    'categories': len(coco_result.get('categories', []))
                }
            }
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500