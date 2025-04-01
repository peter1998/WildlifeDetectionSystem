import os
from flask import Flask
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

# Initialize SQLAlchemy
db = SQLAlchemy()

def create_app(test_config=None):
    # Create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    
    # Set up CORS to allow requests from frontend
    CORS(app)
    
    # Configure the app
    app.config.from_mapping(
        SECRET_KEY='dev',
        SQLALCHEMY_DATABASE_URI='sqlite:///' + os.path.join(app.instance_path, 'wildlife.sqlite'),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        UPLOAD_FOLDER=os.path.abspath('../data/raw_images'),
        PROCESSED_FOLDER=os.path.abspath('../data/processed_images'),
        ANNOTATION_FOLDER=os.path.abspath('../data/annotations'),
        MODEL_FOLDER=os.path.abspath('../models'),
    )

    if test_config is None:
        # Load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # Load the test config if passed in
        app.config.from_mapping(test_config)

    # Ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass
    
    # Initialize database with the app
    db.init_app(app)
    
    # Register blueprints (routes)
    from app.api import images, annotations, models as model_routes
    app.register_blueprint(images.bp)
    app.register_blueprint(annotations.bp)
    app.register_blueprint(model_routes.bp)
    
    return app
