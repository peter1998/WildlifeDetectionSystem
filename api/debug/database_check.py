import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from app import create_app, db
from app.models.models import Species, Image, Annotation
from sqlalchemy import inspect

def detailed_database_diagnostics():
    print("=== Database Detailed Diagnostics ===")
    
    app = create_app('development')
    
    with app.app_context():
        # Inspect database tables
        inspector = inspect(db.engine)
        
        print("\n1. Database Tables:")
        for table_name in inspector.get_table_names():
            print(f"   - {table_name}")
            
            # Column details
            columns = inspector.get_columns(table_name)
            for column in columns:
                print(f"     * {column['name']}: {column['type']}")
        
        # Sample data retrieval
        print("\n2. Sample Data:")
        
        # Species
        print("   Species Sample:")
        species_sample = Species.query.limit(5).all()
        for s in species_sample:
            print(f"     - {s.name} (ID: {s.id}, Scientific Name: {s.scientific_name})")
        
        # Images
        print("\n   Images Sample:")
        images_sample = Image.query.limit(5).all()
        for img in images_sample:
            print(f"     - {img.filename} (ID: {img.id}, Upload Date: {img.upload_date})")
        
        # Annotations
        print("\n   Annotations Sample:")
        annotations_sample = Annotation.query.limit(5).all()
        for ann in annotations_sample:
            print(f"     - Image ID: {ann.image_id}, Species ID: {ann.species_id}")

if __name__ == '__main__':
    detailed_database_diagnostics()