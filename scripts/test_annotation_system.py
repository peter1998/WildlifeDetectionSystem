#!/usr/bin/env python3
# scripts/test_annotation_system.py

import os
import sys
import json
import requests
from pathlib import Path

# Add the parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from our app
from api.app import create_app
from api.app.models import db, Species, Image, Annotation

def test_species_initialization():
    """Test initializing default species."""
    print("Testing species initialization...")
    
    response = requests.post('http://localhost:5000/api/species/initialize')
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Check what species we have
    response = requests.get('http://localhost:5000/api/species/')
    print(f"Total species: {len(response.json()['species'])}")
    for species in response.json()['species']:
        print(f"- {species['name']}")
    
    print("\n")

def test_image_indexing():
    """Test indexing existing images."""
    print("Testing image indexing...")
    
    response = requests.post('http://localhost:5000/api/images/index-existing')
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Check how many images we have
    response = requests.get('http://localhost:5000/api/images/')
    print(f"Total images: {response.json()['total']}")
    print("\n")

def test_folders():
    """Test retrieving folders."""
    print("Testing folder retrieval...")
    
    response = requests.get('http://localhost:5000/api/images/folders')
    print(f"Status code: {response.status_code}")
    
    if response.status_code == 200:
        folders = response.json()['folders']
        print(f"Total folders: {len(folders)}")
        for folder in folders:
            print(f"- {folder}")
    else:
        print(f"Error: {response.text}")
    
    print("\n")

def test_create_annotation():
    """Test creating an annotation."""
    print("Testing annotation creation...")
    
    # Get the first image
    response = requests.get('http://localhost:5000/api/images/')
    if response.status_code == 200 and response.json()['total'] > 0:
        image = response.json()['images'][0]
        image_id = image['id']
        
        # Get the first species
        response = requests.get('http://localhost:5000/api/species/')
        if response.status_code == 200 and len(response.json()['species']) > 0:
            species = response.json()['species'][0]
            species_id = species['id']
            
            # Create an annotation
            annotation_data = {
                'image_id': image_id,
                'species_id': species_id,
                'x_min': 0.1,
                'y_min': 0.1,
                'x_max': 0.5,
                'y_max': 0.5,
                'is_verified': True
            }
            
            response = requests.post(
                'http://localhost:5000/api/annotations/',
                json=annotation_data
            )
            
            print(f"Status code: {response.status_code}")
            print(f"Response: {response.json()}")
            
            # Verify the annotation exists
            response = requests.get(f'http://localhost:5000/api/annotations/image/{image_id}')
            print(f"Annotations for image {image_id}: {response.json()}")
        else:
            print("No species found.")
    else:
        print("No images found.")
    
    print("\n")

def test_batch_annotations():
    """Test creating batch annotations."""
    print("Testing batch annotation creation...")
    
    # Get the first image
    response = requests.get('http://localhost:5000/api/images/')
    if response.status_code == 200 and response.json()['total'] > 0:
        image = response.json()['images'][0]
        image_id = image['id']
        
        # Get species
        response = requests.get('http://localhost:5000/api/species/')
        if response.status_code == 200 and len(response.json()['species']) > 0:
            species = response.json()['species']
            
            # Create batch annotations
            batch_data = {
                'image_id': image_id,
                'annotations': [
                    {
                        'species_id': species[0]['id'],
                        'x_min': 0.1,
                        'y_min': 0.1,
                        'x_max': 0.5,
                        'y_max': 0.5,
                        'is_verified': True
                    },
                    {
                        'species_id': species[1]['id'] if len(species) > 1 else species[0]['id'],
                        'x_min': 0.6,
                        'y_min': 0.6,
                        'x_max': 0.9,
                        'y_max': 0.9,
                        'is_verified': True
                    }
                ]
            }
            
            response = requests.post(
                'http://localhost:5000/api/annotations/batch',
                json=batch_data
            )
            
            print(f"Status code: {response.status_code}")
            print(f"Response: {response.json()}")
        else:
            print("No species found.")
    else:
        print("No images found.")
    
    print("\n")

def test_yolo_export():
    """Test exporting annotations in YOLO format."""
    print("Testing YOLO export...")
    
    export_dir = os.path.expanduser("~/Desktop/TU PHD/WildlifeDetectionSystem/data/export/yolo_test")
    
    response = requests.get(
        'http://localhost:5000/api/annotations/export/yolo',
        params={'output_dir': export_dir}
    )
    
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    if response.status_code == 200 and response.json()['success']:
        print(f"YOLO export saved to: {export_dir}")
        print(f"Images: {response.json()['images_count']}")
        print(f"Classes: {response.json()['classes_count']}")
        
        # Check the files
        if os.path.exists(export_dir):
            print("Files created:")
            for root, dirs, files in os.walk(export_dir):
                for file in files:
                    print(f"- {os.path.join(root, file)}")
    
    print("\n")

def main():
    print("=== Wildlife Detection System Test ===\n")
    
    # Make sure the server is running
    try:
        response = requests.get('http://localhost:5000/health')
        if response.status_code != 200:
            print("Server not responding correctly. Make sure it's running on port 5000.")
            return
    except requests.exceptions.ConnectionError:
        print("Could not connect to server. Make sure it's running on port 5000.")
        return
    
    # Run tests
    test_species_initialization()
    test_image_indexing()
    test_folders()
    test_create_annotation()
    test_batch_annotations()
    test_yolo_export()
    
    print("=== All tests completed ===")

if __name__ == "__main__":
    main()