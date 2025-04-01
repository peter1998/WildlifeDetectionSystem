#!/usr/bin/env python3
"""
Batch annotation tool to quickly annotate multiple images with similar patterns.
This is especially useful for sequences where the same animal appears in multiple frames.
"""

import os
import sys
import json
import argparse

# Ensure we can import from the app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.models.models import db, Image, Annotation, Species
from app.services.annotation_service import AnnotationService
from app.services.image_service import ImageService

def list_species():
    """List all available species in the database."""
    species_list = Species.query.all()
    print("Available species:")
    for species in species_list:
        print(f"  {species.id}: {species.name}")

def list_folders():
    """List all available image folders."""
    folders = ImageService.get_folders()
    print("Available folders:")
    for folder in folders:
        count = Image.query.filter(Image.filename.like(f"{folder}%")).count()
        print(f"  {folder} ({count} images)")

def list_images(folder, limit=20, offset=0, has_annotations=None):
    """List images in the specified folder."""
    query = Image.query.filter(Image.filename.like(f"{folder}%"))
    
    if has_annotations is not None:
        if has_annotations:
            query = query.join(Annotation).group_by(Image.id)
        else:
            # Images with no annotations
            query = query.outerjoin(Annotation).group_by(Image.id).having(db.func.count(Annotation.id) == 0)
    
    total = query.count()
    images = query.order_by(Image.id).limit(limit).offset(offset).all()
    
    print(f"Images in {folder} (showing {offset+1}-{min(offset+limit, total)} of {total}):")
    for i, image in enumerate(images):
        annotation_count = Annotation.query.filter_by(image_id=image.id).count()
        print(f"  {i+1+offset}: {image.filename} ({annotation_count} annotations)")
    
    print(f"\nUse --offset {offset+limit} to see the next page")
    return images

def create_batch_annotation(folder, species_id, x_min, y_min, x_max, y_max, 
                           start_image=None, count=10, confidence=1.0):
    """
    Create the same annotation for multiple consecutive images.
    Useful for tracking the same animal across a sequence.
    """
    # Validate species
    species = Species.query.get(species_id)
    if not species:
        print(f"Error: Species with ID {species_id} not found")
        return
    
    # Get images to annotate
    if start_image:
        # Find the starting image
        start_img = Image.query.filter_by(filename=start_image).first()
        if not start_img:
            print(f"Error: Starting image {start_image} not found")
            return
        
        # Get subsequent images in the same folder
        folder_prefix = os.path.dirname(start_img.filename)
        images = Image.query.filter(
            Image.filename.like(f"{folder_prefix}%"),
            Image.id >= start_img.id
        ).order_by(Image.id).limit(count).all()
    else:
        # Get images from the specified folder
        images = Image.query.filter(
            Image.filename.like(f"{folder}%")
        ).order_by(Image.id).limit(count).all()
    
    # Create annotations
    created_count = 0
    for image in images:
        try:
            # Check if annotation already exists
            existing = Annotation.query.filter_by(
                image_id=image.id,
                species_id=species_id,
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max
            ).first()
            
            if existing:
                print(f"Skipping {image.filename} - annotation already exists")
                continue
            
            # Create the annotation
            annotation = Annotation(
                image_id=image.id,
                species_id=species_id,
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                confidence=confidence,
                is_verified=True
            )
            
            db.session.add(annotation)
            db.session.commit()
            created_count += 1
            print(f"Created annotation for {image.filename}")
            
        except Exception as e:
            print(f"Error creating annotation for {image.filename}: {str(e)}")
            db.session.rollback()
    
    print(f"Created {created_count} new annotations")

def main():
    parser = argparse.ArgumentParser(description="Batch annotation tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List species command
    subparsers.add_parser("species", help="List all available species")
    
    # List folders command
    subparsers.add_parser("folders", help="List all available image folders")
    
    # List images command
    list_parser = subparsers.add_parser("images", help="List images in a folder")
    list_parser.add_argument("folder", help="Folder name")
    list_parser.add_argument("--limit", type=int, default=20, help="Number of images to list")
    list_parser.add_argument("--offset", type=int, default=0, help="Offset for pagination")
    list_parser.add_argument("--annotated", action="store_true", help="Show only annotated images")
    list_parser.add_argument("--unannotated", action="store_true", help="Show only unannotated images")
    
    # Batch annotation command
    batch_parser = subparsers.add_parser("annotate", help="Create batch annotations")
    batch_parser.add_argument("folder", help="Folder name")
    batch_parser.add_argument("species_id", type=int, help="Species ID")
    batch_parser.add_argument("--x_min", type=float, required=True, help="Left boundary (0-1)")
    batch_parser.add_argument("--y_min", type=float, required=True, help="Top boundary (0-1)")
    batch_parser.add_argument("--x_max", type=float, required=True, help="Right boundary (0-1)")
    batch_parser.add_argument("--y_max", type=float, required=True, help="Bottom boundary (0-1)")
    batch_parser.add_argument("--start", help="Starting image filename")
    batch_parser.add_argument("--count", type=int, default=10, help="Number of images to annotate")
    batch_parser.add_argument("--confidence", type=float, default=1.0, help="Annotation confidence")
    
    args = parser.parse_args()
    
    app = create_app()
    with app.app_context():
        if args.command == "species":
            list_species()
        elif args.command == "folders":
            list_folders()
        elif args.command == "images":
            has_annotations = None
            if args.annotated:
                has_annotations = True
            elif args.unannotated:
                has_annotations = False
            
            list_images(args.folder, args.limit, args.offset, has_annotations)
        elif args.command == "annotate":
            create_batch_annotation(
                args.folder, 
                args.species_id,
                args.x_min,
                args.y_min,
                args.x_max,
                args.y_max,
                args.start,
                args.count,
                args.confidence
            )
        else:
            parser.print_help()

if __name__ == "__main__":
    main()