import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from app import create_app, db
from app.models.models import Species, Image, Annotation

def check_app_connectivity():
    print("=== Application Connectivity Diagnostics ===")
    
    try:
        # Create app in development mode
        app = create_app('development')
        
        # Check basic app configuration
        print("\n1. Application Configuration:")
        print(f"   Debug Mode: {app.debug}")
        print(f"   Database URI: {app.config['SQLALCHEMY_DATABASE_URI']}")
        
        # Check database connection
        with app.app_context():
            try:
                # Test database connection
                db.session.execute('SELECT 1')
                print("\n2. Database Connection: ✅ Successful")
                
                # Count records
                print("\n3. Database Record Counts:")
                print(f"   Species: {Species.query.count()}")
                print(f"   Images: {Image.query.count()}")
                print(f"   Annotations: {Annotation.query.count()}")
                
            except Exception as db_error:
                print(f"\n2. Database Connection: ❌ Failed")
                print(f"   Error: {db_error}")
        
        # Check extensions
        print("\n4. Registered Extensions:")
        for ext_name in app.extensions:
            print(f"   - {ext_name}")
        
    except Exception as app_error:
        print(f"Critical Application Error: {app_error}")

if __name__ == '__main__':
    check_app_connectivity()