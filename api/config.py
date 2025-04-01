import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Get absolute path to the base directory
basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    """Base configuration class."""
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key'
    DEBUG = False
    TESTING = False
    
    # Ensure instance directory exists
    INSTANCE_PATH = os.path.join(basedir, 'instance')
    os.makedirs(INSTANCE_PATH, exist_ok=True)
    
    # SQLAlchemy settings with absolute path
    database_path = os.path.join(INSTANCE_PATH, 'wildlife_detection.db')
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or f'sqlite:///{database_path}'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # File storage settings
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw_images')
    PROCESSED_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'processed_images')
    ANNOTATIONS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'annotations')
    EXPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'export')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload size
    
    # Model settings
    MODEL_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'trained')
    CONFIG_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'configs')
    
    # Environmental analysis settings
    LIGHT_CONDITIONS = ['Full darkness', 'Early twilight', 'Late twilight', 'Daylight']
    MOON_PHASES = ['New', 'Waxing Crescent', 'First Quarter', 'Waxing Gibbous', 
                   'Full', 'Waning Gibbous', 'Last Quarter', 'Waning Crescent']
    HABITAT_TYPES = ['Plains', 'Mountains', 'Forest', 'Mixed']
    
    # Behavioral tracking settings
    PREDATOR_SPECIES = ['Wolf', 'Fox', 'Jackal', 'Brown Bear']
    PREY_SPECIES = ['Red Deer', 'Male Roe Deer', 'Female Roe Deer', 'Rabbit', 'Hare']
    BEHAVIOR_TYPES = ['Feeding', 'Resting', 'Moving', 'Alert', 'Interaction', 'Other']


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    

class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    test_db_path = os.path.join(Config.INSTANCE_PATH, 'test_wildlife_detection.db')
    SQLALCHEMY_DATABASE_URI = f'sqlite:///{test_db_path}'
    

class ProductionConfig(Config):
    """Production configuration."""
    # Production database (could be PostgreSQL)
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}