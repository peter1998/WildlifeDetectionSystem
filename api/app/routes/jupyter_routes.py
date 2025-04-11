import os
import subprocess
import threading
import time
import webbrowser
from flask import Blueprint, jsonify, current_app
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Create blueprint
jupyter_bp = Blueprint('jupyter_bp', __name__, url_prefix='/api/jupyter')

# Global variables for tracking Jupyter process
jupyter_process = None
jupyter_url = "http://localhost:8888/lab"

def get_notebook_paths():
    """Get paths to notebook directories and training notebook."""
    # Use project root directory structure
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    notebook_dir = os.path.join(project_root, "notebooks")
    training_notebook = os.path.join(notebook_dir, "training", "wildlife_model.ipynb")
    
    logger.info(f"Project root: {project_root}")
    logger.info(f"Notebook directory: {notebook_dir}")
    logger.info(f"Training notebook: {training_notebook}")
    
    return notebook_dir, training_notebook

def start_jupyter_lab():
    """Start Jupyter Lab as a separate process if not already running."""
    global jupyter_process
    
    if jupyter_process and jupyter_process.poll() is None:
        # Process is already running
        logger.info("Jupyter Lab process is already running")
        return True
    
    try:
        # Get notebook paths
        notebook_dir, training_notebook = get_notebook_paths()
        
        # Make sure directories exist
        os.makedirs(os.path.dirname(training_notebook), exist_ok=True)
        
        logger.info(f"Starting Jupyter Lab with notebook directory: {notebook_dir}")
        
        # Start Jupyter Lab in the notebooks folder
        jupyter_process = subprocess.Popen(
            ["jupyter", "lab", "--no-browser", "--port=8888", "--notebook-dir=" + notebook_dir],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for Jupyter to fully start
        time.sleep(3)
        
        is_running = jupyter_process.poll() is None
        if is_running:
            logger.info("Jupyter Lab started successfully")
        else:
            logger.error("Failed to start Jupyter Lab")
            
        return is_running
    except Exception as e:
        logger.error(f"Error starting Jupyter Lab: {e}")
        return False

@jupyter_bp.route('/start', methods=['GET'])
def start_jupyter():
    """API endpoint for starting Jupyter Lab."""
    success = start_jupyter_lab()
    
    if success:
        # Get notebook paths
        notebook_dir, training_notebook = get_notebook_paths()
        
        # Prepare URL with direct link to the notebook (if exists)
        notebook_url = jupyter_url
        if os.path.exists(training_notebook):
            relative_path = os.path.relpath(training_notebook, notebook_dir)
            # Fix Windows paths for URL
            relative_path = relative_path.replace('\\', '/')
            notebook_url = f"{jupyter_url}/tree/{relative_path}"
            logger.info(f"Opening notebook: {notebook_url}")
        
        return jsonify({
            "success": True,
            "message": "Jupyter Lab started successfully",
            "url": notebook_url
        })
    else:
        return jsonify({
            "success": False,
            "message": "Failed to start Jupyter Lab. Check if Jupyter is installed."
        }), 500

@jupyter_bp.route('/status', methods=['GET'])
def jupyter_status():
    """Check the status of the Jupyter process."""
    is_running = jupyter_process is not None and jupyter_process.poll() is None
    return jsonify({
        "running": is_running, 
        "url": jupyter_url if is_running else None
    })