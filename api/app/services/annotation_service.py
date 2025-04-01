from flask import current_app
import os
import json
import shutil
import yaml
from app import db
from app.models.models import Image, Annotation, Species  # Updated import path
from datetime import datetime

class AnnotationService:
    @staticmethod
    def get_annotations_by_image_id(image_id):
        """Get all annotations for a specific image."""
        annotations = Annotation.query.filter_by(image_id=image_id).all()
        return annotations
    
    @staticmethod
    def get_annotation_by_id(annotation_id):
        """Get an annotation by its ID."""
        return Annotation.query.get_or_404(annotation_id)
    
    @staticmethod
    def create_annotation(image_id, species_id, x_min, y_min, x_max, y_max, confidence=None, is_verified=False):
        """Create a new annotation for an image."""
        # Validate image exists
        image = Image.query.get_or_404(image_id)
        
        # Validate species exists
        species = Species.query.get_or_404(species_id)
        
        # Validate coordinates (must be between 0-1)
        if not (0 <= x_min <= 1 and 0 <= y_min <= 1 and 0 <= x_max <= 1 and 0 <= y_max <= 1):
            raise ValueError("Coordinates must be normalized (0-1)")
        
        # Create the annotation
        annotation = Annotation(
            image_id=image_id,
            species_id=species_id,
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
            confidence=confidence,
            is_verified=is_verified
        )
        
        db.session.add(annotation)
        db.session.commit()
        
        return annotation
    
    @staticmethod
    def create_batch_annotations(image_id, annotations_data):
        """Create multiple annotations for an image at once."""
        # First delete any existing annotations for this image
        Annotation.query.filter_by(image_id=image_id).delete()
        
        created_annotations = []
        for annotation_data in annotations_data:
            annotation = Annotation(
                image_id=image_id,
                species_id=annotation_data['species_id'],
                x_min=annotation_data['x_min'],
                y_min=annotation_data['y_min'],
                x_max=annotation_data['x_max'],
                y_max=annotation_data['y_max'],
                confidence=annotation_data.get('confidence'),
                is_verified=annotation_data.get('is_verified', False)
            )
            db.session.add(annotation)
            created_annotations.append(annotation)
        
        db.session.commit()
        return created_annotations
    
    @staticmethod
    def update_annotation(annotation_id, species_id=None, x_min=None, y_min=None, x_max=None, y_max=None, 
                         confidence=None, is_verified=None):
        """Update an existing annotation."""
        annotation = Annotation.query.get_or_404(annotation_id)
        
        if species_id is not None:
            # Validate species exists
            Species.query.get_or_404(species_id)
            annotation.species_id = species_id
            
        if x_min is not None:
            annotation.x_min = x_min
        if y_min is not None:
            annotation.y_min = y_min
        if x_max is not None:
            annotation.x_max = x_max
        if y_max is not None:
            annotation.y_max = y_max
        if confidence is not None:
            annotation.confidence = confidence
        if is_verified is not None:
            annotation.is_verified = is_verified
            
        annotation.updated_at = datetime.utcnow()
        db.session.commit()
        
        return annotation
    
    @staticmethod
    def delete_annotation(annotation_id):
        """Delete an annotation."""
        annotation = Annotation.query.get_or_404(annotation_id)
        db.session.delete(annotation)
        db.session.commit()
        
        return True
    
    @staticmethod
    def export_coco_format(output_path=None):
        """
        Export all annotations in COCO format.
        
        Args:
            output_path (str, optional): Path to save the COCO JSON file
        
        Returns:
            dict: COCO format JSON
        """
        # Get all species
        species = Species.query.all()
        species_map = {s.id: i+1 for i, s in enumerate(species)}  # COCO uses 1-indexed categories
        
        # Create categories list
        categories = []
        for s in species:
            categories.append({
                'id': species_map[s.id],
                'name': s.name,
                'supercategory': 'wildlife'
            })
        
        # Get all images with annotations
        images_with_annotations = db.session.query(Image).join(Annotation).distinct().all()
        
        # Create images and annotations lists
        images = []
        annotations = []
        annotation_id = 1  # COCO uses unique IDs for annotations
        
        for img in images_with_annotations:
            # Add image info
            image_info = {
                'id': img.id,
                'file_name': os.path.basename(img.filename),
                'width': img.width or 0,
                'height': img.height or 0,
                'date_captured': img.timestamp.isoformat() if img.timestamp else None
            }
            images.append(image_info)
            
            # Add annotations for this image
            for ann in img.annotations:
                # Convert normalized coordinates to absolute pixel coordinates
                width = img.width or 0
                height = img.height or 0
                
                x_min = ann.x_min * width
                y_min = ann.y_min * height
                x_max = ann.x_max * width
                y_max = ann.y_max * height
                
                x = x_min
                y = y_min
                w = x_max - x_min
                h = y_max - y_min
                
                area = w * h
                
                # Create annotation object
                anno = {
                    'id': annotation_id,
                    'image_id': img.id,
                    'category_id': species_map[ann.species_id],
                    'bbox': [x, y, w, h],
                    'area': area,
                    'segmentation': [],  # We don't have segmentation data
                    'iscrowd': 0
                }
                
                annotations.append(anno)
                annotation_id += 1
        
        # Create COCO format JSON
        coco_data = {
            'info': {
                'description': 'Wildlife Camera Trap Dataset',
                'url': '',
                'version': '1.0',
                'year': datetime.now().year,
                'contributor': '',
                'date_created': datetime.now().isoformat()
            },
            'licenses': [{
                'id': 1,
                'name': 'Unknown',
                'url': ''
            }],
            'categories': categories,
            'images': images,
            'annotations': annotations
        }
        
        # Save to file if output_path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(coco_data, f)
        
        return coco_data
    
    @staticmethod
    def export_yolo_format(output_dir):
        """
        Export annotations in YOLO format for training.
        
        Args:
            output_dir (str): Directory to save YOLO format files
            
        Returns:
            dict: Summary of exported files
        """
        # Create output directories
        images_dir = os.path.join(output_dir, 'images')
        labels_dir = os.path.join(output_dir, 'labels')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        # Get all species for class mapping
        species = Species.query.all()
        species_map = {s.id: i for i, s in enumerate(species)}
        
        # Write classes.txt
        with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
            for s in sorted(species, key=lambda x: species_map[x.id]):
                f.write(f"{s.name}\n")
        
        # Process each image with annotations
        images_with_annotations = db.session.query(Image).join(Annotation).distinct().all()
        
        for img in images_with_annotations:
            # Copy image to images directory
            shutil.copy(img.original_path, os.path.join(images_dir, os.path.basename(img.filename)))
            
            # Create label file
            label_file = os.path.join(labels_dir, os.path.basename(img.filename).rsplit('.', 1)[0] + '.txt')
            
            with open(label_file, 'w') as f:
                for ann in img.annotations:
                    # Convert to YOLO format: class x_center y_center width height
                    class_id = species_map[ann.species_id]
                    x_center = (ann.x_min + ann.x_max) / 2
                    y_center = (ann.y_min + ann.y_max) / 2
                    width = ann.x_max - ann.x_min
                    height = ann.y_max - ann.y_min
                    
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
        
        # Create data.yaml
        with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
            f.write(f"train: {os.path.abspath(images_dir)}\n")
            f.write(f"val: {os.path.abspath(images_dir)}\n")  # Use same for now
            f.write(f"nc: {len(species)}\n")
            f.write("names: [")
            for i, s in enumerate(sorted(species, key=lambda x: species_map[x.id])):
                if i > 0:
                    f.write(", ")
                f.write(f"'{s.name}'")
            f.write("]\n")
        
        return {
            'success': True,
            'images_count': len(images_with_annotations),
            'classes_count': len(species),
            'output_dir': output_dir
        }
