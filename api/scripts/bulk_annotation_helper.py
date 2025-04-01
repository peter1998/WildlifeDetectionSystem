#!/usr/bin/env python3
"""
Bulk annotation helper for Wildlife Detection System

This script helps with efficiently annotating large batches of camera trap images.
It identifies images most likely to contain animals and helps prioritize the labeling work.
"""

import os
import sys
import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path

# Ensure we can import from the app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.models.models import db, Image, Annotation, Species
from app.services.image_service import ImageService

def get_unannotated_images(folder=None, limit=None):
    """Get images that haven't been annotated yet."""
    query = """
    SELECT i.id, i.filename, i.width, i.height 
    FROM image i
    LEFT JOIN annotation a ON i.id = a.image_id
    WHERE a.id IS NULL
    """
    
    if folder:
        query += f" AND i.filename LIKE '{folder}%'"
    
    query += " ORDER BY i.id"
    
    if limit:
        query += f" LIMIT {limit}"
    
    conn = sqlite3.connect('instance/wildlife_detection.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(query)
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return results

def get_animal_species():
    """Get all species excluding 'Background'."""
    conn = sqlite3.connect('instance/wildlife_detection.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM species WHERE name != 'Background' ORDER BY name")
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return results

def create_annotation_for_image(image_id, species_id, x_min, y_min, x_max, y_max):
    """Create a new annotation for an image."""
    app = create_app()
    with app.app_context():
        # Check if image exists
        image = Image.query.get(image_id)
        if not image:
            print(f"Error: Image with ID {image_id} not found")
            return False
        
        # Check if species exists
        species = Species.query.get(species_id)
        if not species:
            print(f"Error: Species with ID {species_id} not found")
            return False
        
        # Create annotation
        annotation = Annotation(
            image_id=image_id,
            species_id=species_id,
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
            is_verified=True
        )
        
        try:
            db.session.add(annotation)
            db.session.commit()
            print(f"Created annotation for image {image.filename} with species {species.name}")
            return True
        except Exception as e:
            db.session.rollback()
            print(f"Error creating annotation: {str(e)}")
            return False

def mark_as_no_animals(image_id):
    """Mark an image as having no animals (full-image Background annotation)."""
    app = create_app()
    with app.app_context():
        # Check if image exists
        image = Image.query.get(image_id)
        if not image:
            print(f"Error: Image with ID {image_id} not found")
            return False
        
        # Find Background species
        background = Species.query.filter_by(name='Background').first()
        if not background:
            print("Error: Background species not found")
            return False
        
        # Check if annotation already exists
        existing = Annotation.query.filter_by(
            image_id=image_id,
            species_id=background.id,
            x_min=0,
            y_min=0,
            x_max=1,
            y_max=1
        ).first()
        
        if existing:
            print(f"Image {image.filename} already marked as having no animals")
            return True
        
        # Create full-image background annotation
        annotation = Annotation(
            image_id=image_id,
            species_id=background.id,
            x_min=0,
            y_min=0,
            x_max=1,
            y_max=1,
            is_verified=True
        )
        
        try:
            db.session.add(annotation)
            db.session.commit()
            print(f"Marked image {image.filename} as having no animals")
            return True
        except Exception as e:
            db.session.rollback()
            print(f"Error marking image as having no animals: {str(e)}")
            return False

def batch_annotate_sequence(folder, start_image_id, end_image_id, species_id, box_coords):
    """Annotate a sequence of images with the same bounding box and species."""
    app = create_app()
    with app.app_context():
        # Check if species exists
        species = Species.query.get(species_id)
        if not species:
            print(f"Error: Species with ID {species_id} not found")
            return False
        
        # Get images in the sequence
        images = Image.query.filter(
            Image.id >= start_image_id,
            Image.id <= end_image_id,
            Image.filename.like(f"{folder}%")
        ).order_by(Image.id).all()
        
        if not images:
            print(f"Error: No images found in the specified range")
            return False
        
        print(f"Found {len(images)} images in the sequence")
        
        # Extract box coordinates
        x_min, y_min, x_max, y_max = box_coords
        
        # Create annotations for each image
        success_count = 0
        for image in images:
            try:
                # Check if annotation already exists
                existing = Annotation.query.filter_by(
                    image_id=image.id,
                    species_id=species_id
                ).first()
                
                if existing:
                    print(f"Skipping {image.filename} - annotation already exists")
                    continue
                
                # Create annotation
                annotation = Annotation(
                    image_id=image.id,
                    species_id=species_id,
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                    is_verified=True
                )
                
                db.session.add(annotation)
                db.session.commit()
                
                success_count += 1
                print(f"Created annotation for {image.filename}")
            
            except Exception as e:
                db.session.rollback()
                print(f"Error creating annotation for {image.filename}: {str(e)}")
        
        print(f"Successfully annotated {success_count} out of {len(images)} images")
        return success_count > 0

def bulk_mark_no_animals(folder, start_image_id, end_image_id):
    """Mark a range of images as having no animals."""
    app = create_app()
    with app.app_context():
        # Find Background species
        background = Species.query.filter_by(name='Background').first()
        if not background:
            print("Error: Background species not found")
            return False
        
        # Get images in the range
        images = Image.query.filter(
            Image.id >= start_image_id,
            Image.id <= end_image_id,
            Image.filename.like(f"{folder}%")
        ).order_by(Image.id).all()
        
        if not images:
            print(f"Error: No images found in the specified range")
            return False
        
        print(f"Found {len(images)} images in the range")
        
        # Mark each image as having no animals
        success_count = 0
        for image in images:
            try:
                # Check if annotation already exists
                existing = Annotation.query.filter_by(image_id=image.id).first()
                
                if existing:
                    print(f"Skipping {image.filename} - annotation already exists")
                    continue
                
                # Create full-image background annotation
                annotation = Annotation(
                    image_id=image.id,
                    species_id=background.id,
                    x_min=0,
                    y_min=0,
                    x_max=1,
                    y_max=1,
                    is_verified=True
                )
                
                db.session.add(annotation)
                db.session.commit()
                
                success_count += 1
                print(f"Marked {image.filename} as having no animals")
            
            except Exception as e:
                db.session.rollback()
                print(f"Error marking {image.filename} as having no animals: {str(e)}")
        
        print(f"Successfully marked {success_count} out of {len(images)} images as having no animals")
        return success_count > 0

def analyze_annotation_progress():
    """Analyze the current annotation progress."""
    conn = sqlite3.connect('instance/wildlife_detection.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get total image count
    cursor.execute("SELECT COUNT(*) as count FROM image")
    total_images = cursor.fetchone()['count']
    
    # Get annotated image count
    cursor.execute("""
    SELECT COUNT(DISTINCT image_id) as count 
    FROM annotation
    """)
    annotated_images = cursor.fetchone()['count']
    
    # Get count by species
    cursor.execute("""
    SELECT s.name, COUNT(*) as count 
    FROM annotation a
    JOIN species s ON a.species_id = s.id
    GROUP BY s.name
    ORDER BY count DESC
    """)
    species_counts = [dict(row) for row in cursor.fetchall()]
    
    # Get count by folder
    cursor.execute("""
    SELECT 
        SUBSTR(i.filename, 1, INSTR(i.filename, '/') - 1) as folder,
        COUNT(*) as total,
        SUM(CASE WHEN a.id IS NOT NULL THEN 1 ELSE 0 END) as annotated
    FROM image i
    LEFT JOIN annotation a ON i.id = a.image_id
    GROUP BY folder
    ORDER BY folder
    """)
    folder_stats = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    
    print("\n=== Annotation Progress ===")
    print(f"Total images: {total_images}")
    print(f"Annotated images: {annotated_images} ({(annotated_images/total_images*100):.1f}%)")
    print(f"Remaining: {total_images - annotated_images}")
    
    print("\n=== Species Distribution ===")
    for species in species_counts:
        print(f"{species['name']}: {species['count']}")
    
    print("\n=== Folder Progress ===")
    for folder in folder_stats:
        if folder['total'] > 0:
            progress = (folder['annotated'] / folder['total']) * 100
            print(f"{folder['folder']}: {folder['annotated']}/{folder['total']} ({progress:.1f}%)")
    
    return {
        'total_images': total_images,
        'annotated_images': annotated_images,
        'species_counts': species_counts,
        'folder_stats': folder_stats
    }

def find_similar_images(image_id, sensitivity=0.8):
    """
    Find images similar to a given image.
    This could be used to quickly identify sequences of the same animal.
    NOTE: This is a placeholder function - in a real implementation, 
    this would use image hashing or feature extraction.
    """
    app = create_app()
    with app.app_context():
        # Get the reference image
        reference_image = Image.query.get(image_id)
        if not reference_image:
            print(f"Error: Image with ID {image_id} not found")
            return []
        
        # Get filename parts to find similar images
        filename = reference_image.filename
        parts = os.path.basename(filename).split('_')
        
        # In a real implementation, this would use computer vision
        # For now, we'll just find images with similar filenames
        similar_images = Image.query.filter(
            Image.id != image_id,
            Image.filename.like(f"%{'_'.join(parts[1:]) if len(parts) > 1 else parts[0]}%")
        ).order_by(Image.id).limit(10).all()
        
        return [
            {
                'id': img.id,
                'filename': img.filename,
                'width': img.width,
                'height': img.height
            }
            for img in similar_images
        ]

def print_help():
    """Print usage instructions."""
    print("""
Bulk Annotation Helper - Commands:

  list [folder] [limit]
    List unannotated images, optionally filtered by folder and limited to a count
    Example: python bulk_annotation_helper.py list test_01 20

  species
    List all available animal species
    Example: python bulk_annotation_helper.py species

  annotate <image_id> <species_id> <x_min> <y_min> <x_max> <y_max>
    Create an annotation for a single image
    Example: python bulk_annotation_helper.py annotate 123 7 0.2 0.3 0.5 0.7

  noanimals <image_id>
    Mark an image as having no animals
    Example: python bulk_annotation_helper.py noanimals 123

  sequence <folder> <start_id> <end_id> <species_id> <x_min> <y_min> <x_max> <y_max>
    Annotate a sequence of images with the same box and species
    Example: python bulk_annotation_helper.py sequence test_01 100 120 7 0.2 0.3 0.5 0.7

  bulk-noanimals <folder> <start_id> <end_id>
    Mark a range of images as having no animals
    Example: python bulk_annotation_helper.py bulk-noanimals test_01 100 120

  progress
    Show annotation progress statistics
    Example: python bulk_annotation_helper.py progress

  similar <image_id>
    Find similar images (useful for sequences)
    Example: python bulk_annotation_helper.py similar 123
""")

def main():
    if len(sys.argv) < 2:
        print_help()
        return
    
    command = sys.argv[1]
    
    if command == "list":
        folder = sys.argv[2] if len(sys.argv) > 2 else None
        limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
        images = get_unannotated_images(folder, limit)
        print(f"Found {len(images)} unannotated images:")
        for img in images:
            print(f"ID: {img['id']}, Filename: {img['filename']}")
    
    elif command == "species":
        species = get_animal_species()
        print("Available species:")
        for s in species:
            print(f"ID: {s['id']}, Name: {s['name']}")
    
    elif command == "annotate":
        if len(sys.argv) < 8:
            print("Error: Missing arguments")
            print("Usage: python bulk_annotation_helper.py annotate <image_id> <species_id> <x_min> <y_min> <x_max> <y_max>")
            return
        
        image_id = int(sys.argv[2])
        species_id = int(sys.argv[3])
        x_min = float(sys.argv[4])
        y_min = float(sys.argv[5])
        x_max = float(sys.argv[6])
        y_max = float(sys.argv[7])
        
        create_annotation_for_image(image_id, species_id, x_min, y_min, x_max, y_max)
    
    elif command == "noanimals":
        if len(sys.argv) < 3:
            print("Error: Missing image_id")
            print("Usage: python bulk_annotation_helper.py noanimals <image_id>")
            return
        
        image_id = int(sys.argv[2])
        mark_as_no_animals(image_id)
    
    elif command == "sequence":
        if len(sys.argv) < 9:
            print("Error: Missing arguments")
            print("Usage: python bulk_annotation_helper.py sequence <folder> <start_id> <end_id> <species_id> <x_min> <y_min> <x_max> <y_max>")
            return
        
        folder = sys.argv[2]
        start_id = int(sys.argv[3])
        end_id = int(sys.argv[4])
        species_id = int(sys.argv[5])
        box_coords = (float(sys.argv[6]), float(sys.argv[7]), float(sys.argv[8]), float(sys.argv[9]))
        
        batch_annotate_sequence(folder, start_id, end_id, species_id, box_coords)
    
    elif command == "bulk-noanimals":
        if len(sys.argv) < 5:
            print("Error: Missing arguments")
            print("Usage: python bulk_annotation_helper.py bulk-noanimals <folder> <start_id> <end_id>")
            return
        
        folder = sys.argv[2]
        start_id = int(sys.argv[3])
        end_id = int(sys.argv[4])
        
        bulk_mark_no_animals(folder, start_id, end_id)
    
    elif command == "progress":
        analyze_annotation_progress()
    
    elif command == "similar":
        if len(sys.argv) < 3:
            print("Error: Missing image_id")
            print("Usage: python bulk_annotation_helper.py similar <image_id>")
            return
        
        image_id = int(sys.argv[2])
        similar_images = find_similar_images(image_id)
        
        print(f"Found {len(similar_images)} similar images:")
        for img in similar_images:
            print(f"ID: {img['id']}, Filename: {img['filename']}")
    
    else:
        print(f"Unknown command: {command}")
        print_help()

if __name__ == "__main__":
    main()