import os
import sys

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import os
import sys
import sqlite3
import traceback
import stat
from sqlalchemy import text  # Add this import

from app import create_app, db
from app.models.models import Species, Image, Annotation

def detailed_filesystem_check(path):
    """Perform a detailed filesystem check on the given path."""
    print(f"\n=== Filesystem Check for {path} ===")
    try:
        # Basic file/directory existence
        print(f"Path Exists: {os.path.exists(path)}")
        
        if os.path.exists(path):
            # Get file/directory stats
            st = os.stat(path)
            
            # Permissions
            print("Permissions:")
            print(f"  Numeric: {oct(st.st_mode)[-3:]}")
            print(f"  Readable:  {bool(st.st_mode & stat.S_IRUSR)}")
            print(f"  Writable:  {bool(st.st_mode & stat.S_IWUSR)}")
            print(f"  Executable: {bool(st.st_mode & stat.S_IXUSR)}")
            
            # Ownership
            import pwd
            print(f"Owner: {pwd.getpwuid(st.st_uid).pw_name}")
            
            # Additional details
            print(f"Size: {st.st_size} bytes")
            print(f"Last Modified: {os.path.getmtime(path)}")
    except Exception as e:
        print(f"Error checking path {path}: {e}")

def check_database_access():
    print("=== Comprehensive Database Access Diagnostic ===")
    
    try:
        # Create app without admin initialization
        app = create_app('development', init_admin=False)
        
        # Get database path
        db_path = app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', '')
        db_path = os.path.abspath(os.path.join(project_root, db_path))
        
        print(f"\n=== Database Path: {db_path} ===")
        
        # Detailed filesystem check
        detailed_filesystem_check(db_path)
        
        # Check parent directory
        parent_dir = os.path.dirname(db_path)
        detailed_filesystem_check(parent_dir)
        
        # Validate database connection
        with app.app_context():
            try:
                # Direct SQLAlchemy connection test
                print("\n=== SQLAlchemy Connection Test ===")
                db.session.execute(text('SELECT 1'))  # Use text() for SQLAlchemy 2.x
                print("✅ SQLAlchemy connection successful")
            except Exception as sa_error:
                print(f"❌ SQLAlchemy Connection Error: {sa_error}")
                traceback.print_exc()
            
            # Try to count records
            try:
                print("\n=== Record Counting ===")
                print(f"Species Count: {Species.query.count()}")
                print(f"Images Count: {Image.query.count()}")
                print(f"Annotations Count: {Annotation.query.count()}")
            except Exception as count_error:
                print(f"❌ Error counting records: {count_error}")
                traceback.print_exc()
        
        # Direct SQLite connection test
        try:
            print("\n=== Direct SQLite Connection Test ===")
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check existing tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            print("Tables in Database:")
            for table in tables:
                print(f"- {table[0]}")
            
            # Try a sample query
            cursor.execute("SELECT COUNT(*) FROM sqlite_master;")
            table_count = cursor.fetchone()[0]
            print(f"Total Tables: {table_count}")
            
            conn.close()
        except Exception as sqlite_error:
            print(f"❌ SQLite Connection Error: {sqlite_error}")
            traceback.print_exc()
        
    except Exception as overall_error:
        print(f"❌ Overall Diagnostic Error: {overall_error}")
        traceback.print_exc()

    # Additional system-wide checks
    print("\n=== System-wide Checks ===")
    try:
        # Current working directory
        print(f"Current Working Directory: {os.getcwd()}")
        
        # Temporary directory
        import tempfile
        print(f"Temporary Directory: {tempfile.gettempdir()}")
        
        # Python and SQLite versions
        print(f"Python Version: {sys.version}")
        print(f"SQLite Version: {sqlite3.sqlite_version}")
    except Exception as sys_error:
        print(f"Error in system checks: {sys_error}")

if __name__ == '__main__':
    check_database_access()