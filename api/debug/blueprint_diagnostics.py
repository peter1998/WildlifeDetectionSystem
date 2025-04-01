import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from app import create_app

def blueprint_diagnostics():
    print("=== Blueprint Diagnostics ===")
    
    app = create_app('development')
    
    print("\n1. Registered Blueprints:")
    for name, blueprint in app.blueprints.items():
        print(f"   Blueprint: {name}")
        print(f"     Import Name: {blueprint.import_name}")
        print(f"     URL Prefix: {blueprint.url_prefix}")
        
        # Print routes for each blueprint
        print("     Routes:")
        for rule in app.url_map.iter_rules():
            if rule.endpoint.startswith(name):
                print(f"       - {rule.endpoint}: {rule.rule} (Methods: {rule.methods})")
        print()

if __name__ == '__main__':
    blueprint_diagnostics()