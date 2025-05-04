import os
import sys  # Added missing import
import subprocess
import json
import time
import signal
from flask import Blueprint, jsonify, current_app, request
import logging
import psutil  # Make sure to install this: pip install psutil
import shlex  # For properly escaping command arguments

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
jupyter_bp = Blueprint('jupyter_bp', __name__, url_prefix='/api/jupyter')

def is_jupyter_running():
    """Check if Jupyter is already running on port 8888"""
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            cmdline = proc.info.get('cmdline', [])
            if not cmdline:
                continue
                
            cmdline_str = ' '.join(cmdline)
            if ('jupyter-lab' in cmdline_str or 'jupyter lab' in cmdline_str) and '--port=8888' in cmdline_str:
                logger.info(f"Found running Jupyter process: {proc.info.get('pid', 'unknown')}")
                return True, proc.info.get('pid')
        return False, None
    except Exception as e:
        logger.error(f"Error checking for Jupyter process: {e}")
        return False, None

def start_jupyter_lab():
    """Start Jupyter Lab as a separate process if not already running."""
    # First check if already running
    running, pid = is_jupyter_running()
    if running:
        logger.info(f"Jupyter Lab already running with PID {pid}")
        return True, pid
    
    try:
        # Get notebook directory path
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        notebook_dir = os.path.join(project_root, "notebooks")
        
        logger.info(f"Starting Jupyter Lab in directory: {notebook_dir}")
        
        # Try direct launch first (most reliable method)
        try:
            # Construct command with arguments carefully to handle spaces in paths
            cmd = [
                "jupyter", "lab", 
                "--no-browser", 
                "--port=8888", 
                f"--notebook-dir={notebook_dir}"
            ]
            
            # Log the exact command being executed
            logger.info(f"Executing command: {' '.join(cmd)}")
            
            # Start process detached from Flask
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True  # This ensures the process continues when Flask exits
            )
            
            # Wait for Jupyter to start (max 15 seconds)
            for _ in range(15):
                time.sleep(1)
                running, pid = is_jupyter_running()
                if running:
                    logger.info(f"Jupyter Lab started successfully with PID {pid}")
                    return True, pid
                
                # Check if process failed immediately
                if process.poll() is not None:
                    stdout, stderr = process.communicate()
                    logger.error(f"Jupyter failed to start. Return code: {process.returncode}")
                    logger.error(f"STDOUT: {stdout.decode('utf-8', errors='ignore')}")
                    logger.error(f"STDERR: {stderr.decode('utf-8', errors='ignore')}")
                    break
            
            # If we get here, the direct launch method failed or timed out
            logger.warning("Direct Jupyter launch failed or timed out, trying script method...")
        
        except Exception as launch_error:
            logger.error(f"Error during direct Jupyter launch: {launch_error}")
            logger.warning("Falling back to script method...")
        
        # Fallback to script method
        jupyter_cmd = "jupyter"
        
        # Create a starter script that will persist even if Flask restarts
        script_dir = os.path.join(project_root, "temp")
        os.makedirs(script_dir, exist_ok=True)
        
        # Create a log file path
        log_file = os.path.join(script_dir, "jupyter.log")
        
        script_path = os.path.join(script_dir, "start_jupyter.sh")
        with open(script_path, 'w') as f:
            if os.name == 'nt':  # Windows
                f.write(f'start /B "{jupyter_cmd}" lab --no-browser --port=8888 --notebook-dir="{notebook_dir}" > "{log_file}" 2>&1')
            else:  # Linux/Mac
                # Use proper quoting to handle spaces in paths
                f.write(f'#!/bin/bash\n"{jupyter_cmd}" lab --no-browser --port=8888 --notebook-dir="{notebook_dir}" > "{log_file}" 2>&1 &')
        
        # Make script executable on Unix
        if os.name != 'nt':
            os.chmod(script_path, 0o755)
        
        # Start the script
        logger.info(f"Starting Jupyter using script: {script_path}")
        if os.name == 'nt':
            subprocess.Popen(['cmd', '/c', script_path], shell=False)
        else:
            subprocess.Popen(['/bin/bash', script_path], shell=False)
        
        # Wait for Jupyter to start (max 15 seconds)
        for _ in range(15):
            time.sleep(1)
            running, pid = is_jupyter_running()
            if running:
                logger.info(f"Jupyter Lab started successfully with PID {pid}")
                return True, pid
            
            # Check log file for errors
            if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
                with open(log_file, 'r') as f:
                    log_content = f.read()
                    if 'Error:' in log_content or 'Exception:' in log_content:
                        logger.error(f"Found error in Jupyter log: {log_content[-500:]}")  # Last 500 chars
        
        logger.error("Timed out waiting for Jupyter to start")
        return False, None
        
    except Exception as e:
        logger.error(f"Error starting Jupyter Lab: {e}")
        return False, None

@jupyter_bp.route('/start', methods=['GET', 'POST'])
def start_jupyter():
    """API endpoint for starting Jupyter Lab."""
    success, pid = start_jupyter_lab()
    
    if success:
        return jsonify({
            "success": True,
            "message": "Jupyter Lab started successfully",
            "pid": pid,
            "url": "http://localhost:8888/lab"
        })
    else:
        return jsonify({
            "success": False,
            "message": "Failed to start Jupyter Lab. Check logs for details."
        }), 500

@jupyter_bp.route('/status', methods=['GET'])
def jupyter_status():
    """Check the status of the Jupyter process."""
    running, pid = is_jupyter_running()
    return jsonify({
        "running": running, 
        "pid": pid,
        "url": "http://localhost:8888/lab" if running else None
    })

# Add this route to handle opening a specific notebook
@jupyter_bp.route('/open', methods=['POST'])
def open_notebook():
    """Start Jupyter and return URL for specific notebook."""
    # Get notebook path from request
    data = request.get_json()
    notebook_path = data.get('notebook_path', '')
    
    # Make sure Jupyter is running
    success, _ = start_jupyter_lab()
    if not success:
        return jsonify({
            "success": False,
            "message": "Failed to start Jupyter Lab"
        }), 500
        
    # Construct URL for specific notebook
    # For JupyterLab, the URL format should be /lab/tree/path/to/notebook
    base_url = "http://localhost:8888"
    notebook_url = f"{base_url}/lab/tree/{notebook_path.lstrip('/')}"
    
    return jsonify({
        "success": True,
        "message": "Jupyter Lab started successfully",
        "url": notebook_url
    })