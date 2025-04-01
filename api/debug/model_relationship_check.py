import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from app import create_app, db
from app.models.models import Species, Image, Annotation

def check_model_relationships():
    print("=== Model Relationship Diagnostics ===")
    
    app = create_app('development')
    
    with app.app_context():
        # Test Species to Annotations relationship
        print("\n1. Species to Annotations:")
        sample_species = Species.query.first()
        if sample_species:
            print(f"   Species: {sample_species.name}")
            print(f"   Annotations Count: {len(sample_species.annotations)}")
        
        # Test Image to Annotations relationship
        print("\n2. Image to Annotations:")
        sample_image = Image.query.first()
        if sample_image:
            print(f"   Image: {sample_image.filename}")
            print(f"   Annotations Count: {len(sample_image.annotations)}")
        
        # Test Complex Relationships
        print("\n3. Complex Relationship Check:")
        for annotation in Annotation.query.limit(5):
            print(f"   Annotation ID: {annotation.id}")
            print(f"     Species: {annotation.species.name}")
            print(f"     Image: {annotation.image.filename}")

if __name__ == '__main__':
    check_model_relationships()