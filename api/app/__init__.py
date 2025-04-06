from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from config import config
from flask_admin import Admin
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize extensions
db = SQLAlchemy()

def create_app(config_name='development', init_admin=True):
    """
    Application factory function.
    
    Args:
        config_name: The configuration to use (development, testing, production)
        init_admin: Whether to initialize Flask-Admin (default: True)
        
    Returns:
        Flask application instance
    """
    # Create Flask app instance
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object(config[config_name])
    
    # Diagnostic logging of configuration
    logger.info(f"Creating application with config: {config_name}")
    logger.info(f"Database URI: {app.config.get('SQLALCHEMY_DATABASE_URI')}")
    
    # Initialize CORS
    CORS(app)
    
    # Initialize extensions with app
    db.init_app(app)
    
    # Ensure upload directories exist
    directories = [
        'UPLOAD_FOLDER', 
        'PROCESSED_FOLDER', 
        'ANNOTATIONS_FOLDER', 
        'MODEL_FOLDER', 
        'CONFIG_FOLDER'
    ]
    
    for dir_key in directories:
        try:
            dir_path = app.config.get(dir_key)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
                logger.info(f"Created directory: {dir_path}")
            else:
                logger.warning(f"No path configured for {dir_key}")
        except Exception as e:
            logger.error(f"Error creating directory for {dir_key}: {e}")
    
    # Register blueprints with unique names to prevent conflicts
    from .routes.main import main as main_blueprint
    from .routes.images import images as images_blueprint
    from .routes.species import species as species_blueprint
    from .routes.annotations import annotations as annotations_blueprint
    from .routes.static_routes import static_pages as static_pages_blueprint
    from .routes.environmental_routes import environmental as environmental_blueprint
    
    # Register static_pages blueprint first to handle the root URL
    try:
        app.register_blueprint(static_pages_blueprint)
        logger.info("Registered blueprint: static_pages")
    except Exception as e:
        logger.error(f"Failed to register static_pages blueprint: {e}")
    
    # Then register the rest of the blueprints
    blueprint_mappings = [
        (main_blueprint, 'main'),
        (images_blueprint, 'images'),
        (species_blueprint, 'species'),
        (annotations_blueprint, 'annotations'),
        (environmental_blueprint, 'environmental')
    ]
    
    for blueprint, name in blueprint_mappings:
        try:
            app.register_blueprint(blueprint)
            logger.info(f"Registered blueprint: {name}")
        except Exception as e:
            logger.error(f"Failed to register blueprint {name}: {e}")
    
    # Optionally initialize admin
    if init_admin:
        try:
            from .admin import init_admin
            admin = init_admin(app)
            logger.info("Flask-Admin initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Flask-Admin: {e}")
    
    # Create database tables
    with app.app_context():
        try:
            db.create_all()
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
    
    return app