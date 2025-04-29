from flask import current_app
import os
import json
import shutil
import yaml
from app import db
from app.models.models import Image, Annotation, Species
from datetime import datetime
from sqlalchemy import func, distinct
import logging

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
    def get_annotated_datasets():
        """Get list of datasets (folders) that have annotated images."""
        # Get all images that have annotations
        image_folders = db.session.query(
            func.substr(Image.filename, 1, func.instr(Image.filename, '/'))
        ).join(Annotation).group_by(
            func.substr(Image.filename, 1, func.instr(Image.filename, '/'))
        ).all()
        
        # Clean up folder names
        folders = []
        for folder_tuple in image_folders:
            if folder_tuple[0]:
                folder_name = folder_tuple[0].rstrip('/')
                if folder_name:
                    folders.append(folder_name)
        
        # Add statistics for each folder
        datasets = []
        for folder in folders:
            # Count images in folder
            total_images = Image.query.filter(Image.filename.like(f'{folder}/%')).count()
            
            # Count annotated images in folder
            annotated_images = db.session.query(Image.id).filter(
                Image.filename.like(f'{folder}/%')
            ).join(Annotation).distinct().count()
            
            # Get species in folder
            species_in_folder = db.session.query(Species.id, Species.name).join(
                Annotation, Species.id == Annotation.species_id
            ).join(
                Image, Annotation.image_id == Image.id
            ).filter(
                Image.filename.like(f'{folder}/%')
            ).distinct().all()
            
            datasets.append({
                'name': folder,
                'total_images': total_images,
                'annotated_images': annotated_images,
                'completion_percentage': (annotated_images / total_images * 100) if total_images > 0 else 0,
                'species_count': len(species_in_folder),
                'species': [{'id': s.id, 'name': s.name} for s in species_in_folder]
            })
        
        return datasets
    
    @staticmethod
    def export_coco_format(output_dir):
        """
        Export all annotations in COCO format.
        
        Args:
            output_dir (str): Directory to save the COCO JSON file
        
        Returns:
            dict: COCO format JSON
        """
        logging.info(f"Starting COCO export to {output_dir}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
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
        images_list = []
        annotations_list = []
        annotation_id = 1  # COCO uses unique IDs for annotations
        
        # Track image copying progress
        total_images = len(images_with_annotations)
        processed_images = 0
        
        # Create output images directory
        images_dir = os.path.join(output_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        for img in images_with_annotations:
            try:
                # Copy image to output directory
                if img.original_path and os.path.exists(img.original_path):
                    dest_path = os.path.join(images_dir, os.path.basename(img.filename))
                    shutil.copy(img.original_path, dest_path)
                
                # Add image info
                image_info = {
                    'id': img.id,
                    'file_name': os.path.basename(img.filename),
                    'width': img.width or 0,
                    'height': img.height or 0,
                    'date_captured': img.timestamp.isoformat() if img.timestamp else None
                }
                images_list.append(image_info)
                
                # Filter out "Background" annotations
                valid_annotations = []
                for ann in img.annotations:
                    species_name = Species.query.get(ann.species_id).name
                    if species_name.lower() != 'background':
                        valid_annotations.append(ann)
                
                # Add annotations for this image
                for ann in valid_annotations:
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
                    
                    annotations_list.append(anno)
                    annotation_id += 1
                
                processed_images += 1
                if processed_images % 100 == 0:
                    logging.info(f"Processed {processed_images}/{total_images} images for COCO export")
            
            except Exception as e:
                logging.error(f"Error processing image {img.filename}: {str(e)}")
                continue
        
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
            'images': images_list,
            'annotations': annotations_list
        }
        
        # Save to file
        output_path = os.path.join(output_dir, 'annotations.json')
        with open(output_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        logging.info(f"COCO export completed: {output_path}")
        
        return coco_data
    
    @staticmethod
    def export_yolo_format(output_dir, val_split=0.2):
        """
        Export annotations in YOLO format with train/val split.
        
        Args:
            output_dir (str): Directory to save YOLO format files
            val_split (float): Proportion of data to use for validation (0-1)
                
        Returns:
            dict: Summary of exported files
        """
        logging.info(f"Starting YOLO export to {output_dir}")
        
        # Create output directories
        train_images_dir = os.path.join(output_dir, 'images', 'train')
        val_images_dir = os.path.join(output_dir, 'images', 'val')
        train_labels_dir = os.path.join(output_dir, 'labels', 'train')
        val_labels_dir = os.path.join(output_dir, 'labels', 'val')
        
        os.makedirs(train_images_dir, exist_ok=True)
        os.makedirs(val_images_dir, exist_ok=True)
        os.makedirs(train_labels_dir, exist_ok=True)
        os.makedirs(val_labels_dir, exist_ok=True)
        
        # Get all species for class mapping
        species = Species.query.all()
        species_map = {s.id: i for i, s in enumerate(species)}
        
        # Write classes.txt
        with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
            for s in sorted(species, key=lambda x: species_map[x.id]):
                f.write(f"{s.name}\n")
        
        # Get all images with annotations (excluding background-only annotations)
        # First, get images with annotations
        images_with_annotations = db.session.query(Image).join(Annotation).distinct().all()
        
        # Filter to keep only images with non-background annotations
        filtered_images = []
        image_annotation_counts = {}  # Track annotation counts per image
        
        for img in images_with_annotations:
            has_non_background = False
            annotation_count = 0
            
            for ann in img.annotations:
                species_name = Species.query.get(ann.species_id).name
                if species_name.lower() != 'background':
                    has_non_background = True
                    annotation_count += 1
            
            if has_non_background:
                filtered_images.append(img)
                image_annotation_counts[img.id] = annotation_count
        
        # Simple split (but try to distribute annotations evenly)
        import random
        random.seed(42)  # For reproducibility
        
        # Sort by annotation count (descending) to ensure even distribution
        sorted_images = sorted(filtered_images, key=lambda x: image_annotation_counts.get(x.id, 0), reverse=True)
        
        # Alternate between train and val, with appropriate split
        train_images = []
        val_images = []
        
        for i, img in enumerate(sorted_images):
            if i % int(1/val_split) == 0:  # e.g., every 5th image for 20% validation
                val_images.append(img)
            else:
                train_images.append(img)
        
        logging.info(f"Found {len(filtered_images)} images with valid annotations")
        logging.info(f"Using {len(train_images)} for training, {len(val_images)} for validation")
        
        # Simple counters
        train_count = 0
        val_count = 0
        train_ann_count = 0
        val_ann_count = 0
        species_distribution = {}  # Track annotations per species
        
        # Process training images
        for img in train_images:
            if not img.original_path or not os.path.exists(img.original_path):
                logging.warning(f"Image file not found at {img.original_path}")
                continue
                
            try:
                # Copy image to training images directory
                dest_path = os.path.join(train_images_dir, os.path.basename(img.filename))
                shutil.copy(img.original_path, dest_path)
                
                # Create label file
                label_file = os.path.join(train_labels_dir, os.path.basename(img.filename).rsplit('.', 1)[0] + '.txt')
                
                with open(label_file, 'w') as f:
                    annotation_count = 0
                    for ann in img.annotations:
                        species_obj = Species.query.get(ann.species_id)
                        if species_obj.name.lower() == 'background':
                            continue
                            
                        # Convert to YOLO format: class x_center y_center width height
                        class_id = species_map[ann.species_id]
                        x_center = (ann.x_min + ann.x_max) / 2
                        y_center = (ann.y_min + ann.y_max) / 2
                        width = ann.x_max - ann.x_min
                        height = ann.y_max - ann.y_min
                        
                        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
                        annotation_count += 1
                        
                        # Update distribution statistics
                        if species_obj.name not in species_distribution:
                            species_distribution[species_obj.name] = {'train': 0, 'val': 0}
                        species_distribution[species_obj.name]['train'] += 1
                    
                    if annotation_count > 0:
                        train_count += 1
                        train_ann_count += annotation_count
                
            except Exception as e:
                logging.error(f"Error processing training image {img.filename}: {str(e)}")
                continue
        
        # Process validation images
        for img in val_images:
            if not img.original_path or not os.path.exists(img.original_path):
                logging.warning(f"Image file not found at {img.original_path}")
                continue
                
            try:
                # Copy image to validation images directory
                dest_path = os.path.join(val_images_dir, os.path.basename(img.filename))
                shutil.copy(img.original_path, dest_path)
                
                # Create label file
                label_file = os.path.join(val_labels_dir, os.path.basename(img.filename).rsplit('.', 1)[0] + '.txt')
                
                with open(label_file, 'w') as f:
                    annotation_count = 0
                    for ann in img.annotations:
                        species_obj = Species.query.get(ann.species_id)
                        if species_obj.name.lower() == 'background':
                            continue
                            
                        # Convert to YOLO format: class x_center y_center width height
                        class_id = species_map[ann.species_id]
                        x_center = (ann.x_min + ann.x_max) / 2
                        y_center = (ann.y_min + ann.y_max) / 2
                        width = ann.x_max - ann.x_min
                        height = ann.y_max - ann.y_min
                        
                        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
                        annotation_count += 1
                        
                        # Update distribution statistics
                        if species_obj.name not in species_distribution:
                            species_distribution[species_obj.name] = {'train': 0, 'val': 0}
                        species_distribution[species_obj.name]['val'] += 1
                    
                    if annotation_count > 0:
                        val_count += 1
                        val_ann_count += annotation_count
                        
            except Exception as e:
                logging.error(f"Error processing validation image {img.filename}: {str(e)}")
                continue
        
        # Create data.yaml
        with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
            f.write(f"train: {os.path.abspath(train_images_dir)}\n")
            f.write(f"val: {os.path.abspath(val_images_dir)}\n")
            f.write(f"nc: {len(species)}\n")
            
            # Write class names
            f.write("names: [")
            for i, s in enumerate(sorted(species, key=lambda x: species_map[x.id])):
                if i > 0:
                    f.write(", ")
                f.write(f"'{s.name}'")
            f.write("]\n")
        
        # Create detailed dataset report
        report = {
            "dataset_summary": {
                "total_images": len(filtered_images),
                "training_images": train_count,
                "validation_images": val_count,
                "total_annotations": train_ann_count + val_ann_count,
                "training_annotations": train_ann_count,
                "validation_annotations": val_ann_count,
                "species_count": len(species),
                "export_timestamp": datetime.now().isoformat()
            },
            "species_distribution": {}
        }
        
        # Calculate percentages for species distribution
        for species_name, counts in species_distribution.items():
            total = counts['train'] + counts['val']
            report["species_distribution"][species_name] = {
                "train_count": counts['train'],
                "val_count": counts['val'],
                "total_count": total,
                "train_percentage": (counts['train'] / total * 100) if total > 0 else 0,
                "val_percentage": (counts['val'] / total * 100) if total > 0 else 0
            }
        
        # Save report
        with open(os.path.join(output_dir, 'export_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
            
        logging.info(f"YOLO export completed: {output_dir}")
        logging.info(f"Created data.yaml at {os.path.join(output_dir, 'data.yaml')}")
        
        return {
            'success': True,
            'images_count': len(filtered_images),
            'classes_count': len(species),
            'train_images': train_count,
            'val_images': val_count,
            'train_annotations': train_ann_count, 
            'val_annotations': val_ann_count,
            'species_distribution': species_distribution,
            'output_dir': output_dir
        }
    
    @staticmethod
    def get_dataset_stats(dataset_name=None):
        """
        Get statistics for a specific dataset or all datasets.
        
        Args:
            dataset_name (str, optional): Name of the dataset folder
            
        Returns:
            dict: Dataset statistics
        """
        query = db.session.query(Image)
        
        if dataset_name:
            query = query.filter(Image.filename.like(f'{dataset_name}/%'))
        
        # Get total images count
        total_images = query.count()
        
        # Get annotated images count
        annotated_images = query.join(Annotation).distinct().count()
        
        # Get non-background annotations count
        non_background_count = db.session.query(Annotation).join(
            Image
        ).join(
            Species
        ).filter(
            Species.name != 'Background'
        )
        
        if dataset_name:
            non_background_count = non_background_count.filter(Image.filename.like(f'{dataset_name}/%'))
        
        non_background_count = non_background_count.count()
        
        # Get species breakdown
        species_query = db.session.query(
            Species.name, 
            func.count(distinct(Image.id)).label('image_count'),
            func.count(Annotation.id).label('annotation_count')
        ).join(
            Annotation
        ).join(
            Image
        ).filter(
            Species.name != 'Background'
        )
        
        if dataset_name:
            species_query = species_query.filter(Image.filename.like(f'{dataset_name}/%'))
            
        species_stats = species_query.group_by(Species.name).all()
        
        species_breakdown = []
        for name, image_count, annotation_count in species_stats:
            species_breakdown.append({
                'name': name,
                'image_count': image_count,
                'annotation_count': annotation_count
            })
        
        return {
            'dataset_name': dataset_name or 'all',
            'total_images': total_images,
            'annotated_images': annotated_images,
            'completion_percentage': (annotated_images / total_images * 100) if total_images > 0 else 0,
            'total_annotations': non_background_count,
            'species_breakdown': species_breakdown
        }