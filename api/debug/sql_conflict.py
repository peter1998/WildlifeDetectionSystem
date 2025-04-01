import os
import sys

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import sqlite3
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

def precise_database_diagnostic():
    print("=== Precise Database Path and Connection Diagnostic ===")
    
    # Import create_app after path setup
    from app import create_app
    
    # Create app without admin initialization
    app = create_app('development', init_admin=False)
    
    # Get database URI and path
    db_uri = app.config['SQLALCHEMY_DATABASE_URI']
    print(f"Database URI Configuration: {db_uri}")
    
    # Resolve the absolute path
    db_path = db_uri.replace('sqlite:///', '')
    db_path = os.path.abspath(os.path.join(project_root, db_path))
    
    print("\n=== Path Analysis ===")
    print(f"Project Root: {project_root}")
    print(f"Resolved Database Path: {db_path}")
    print(f"File Exists: {os.path.exists(db_path)}")
    
    if os.path.exists(db_path):
        # File permissions and details
        import stat
        st = os.stat(db_path)
        print("\n=== File Details ===")
        print(f"Size: {st.st_size} bytes")
        print(f"Permissions: {oct(st.st_mode)[-3:]}")
        print(f"Readable:  {bool(st.st_mode & stat.S_IRUSR)}")
        print(f"Writable:  {bool(st.st_mode & stat.S_IWUSR)}")
    
    # Direct SQLite connection test
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print("\n=== SQLite Direct Connection ===")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print("Tables in Database:")
        for table in tables:
            print(f"- {table[0]}")
        
        conn.close()
    except Exception as e:
        print(f"\n❌ SQLite Direct Connection Error: {e}")
    
    # SQLAlchemy specific diagnostics
    try:
        engine = create_engine(f'sqlite:///{db_path}', echo=True)
        
        print("\n=== SQLAlchemy Engine Creation ===")
        with engine.connect() as connection:
            result = connection.execute(text('SELECT 1'))
            print("✅ SQLAlchemy Connection Successful")
            print(result.fetchone())
    except Exception as e:
        print(f"\n❌ SQLAlchemy Connection Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    precise_database_diagnostic()