import sys
import os

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import flask
import sqlalchemy

# Now import create_app
from app import create_app

def debug_flask_app():
    print("=== Flask Application Diagnostic ===")
    
    # Verify versions
    print("\nVersions:")
    print(f"Python: {sys.version}")
    print(f"Flask: {flask.__version__}")
    print(f"SQLAlchemy: {sqlalchemy.__version__}")
    
    # Create app
    try:
        app = create_app('development', init_admin=False)
        
        # Use app context for route testing
        with app.app_context():
            print("\n1. Registered Blueprints:")
            for name, blueprint in app.blueprints.items():
                print(f"   - {name}")
                print(f"     URL Prefix: {blueprint.url_prefix}")
            
            print("\n2. Available Routes:")
            for rule in app.url_map.iter_rules():
                print(f"   {rule.endpoint:25s} {rule.rule:30s} {list(rule.methods)}")
            
            # Database configuration
            print("\n3. Database Configuration:")
            print(f"   URI: {app.config.get('SQLALCHEMY_DATABASE_URI')}")
    
    except Exception as e:
        print(f"Error creating application: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    debug_flask_app()