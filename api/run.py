import os
from config import config

# Print debugging information
print(f"Current working directory: {os.getcwd()}")
print(f"Database URI: {config['development'].SQLALCHEMY_DATABASE_URI}")
print(f"Full database path: {os.path.join(os.getcwd(), config['development'].SQLALCHEMY_DATABASE_URI.replace('sqlite:///', ''))}")

# Get environment configuration (default to development)
config_name = os.environ.get('FLASK_CONFIG', 'default')
from app import create_app
app = create_app(config_name)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)