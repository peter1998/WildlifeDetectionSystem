from flask import Blueprint, jsonify, current_app, redirect, url_for
import platform
import sys
import flask
import sqlalchemy

# Create a blueprint for main routes
main = Blueprint('main', __name__)

# Change the root route to redirect to the dashboard
@main.route('/api')
def index():
    """Root endpoint for the API."""
    return jsonify({
        'message': 'Welcome to the Wildlife Detection System API',
        'status': 'operational'
    })

@main.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy'
    })

@main.route('/system-info')
def system_info():
    """Provide system and application diagnostic information."""
    from app import db  # Import db from your app module

    return jsonify({
        'python_version': platform.python_version(),
        'flask_version': flask.__version__,
        'sqlalchemy_version': sqlalchemy.__version__,
        'debug_mode': current_app.debug if current_app else 'N/A',
        'config': str(current_app.config) if current_app else 'N/A',
        'database_uri': current_app.config.get('SQLALCHEMY_DATABASE_URI', 'N/A') if current_app else 'N/A'
    })