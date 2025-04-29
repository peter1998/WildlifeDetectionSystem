from flask import Blueprint, request, jsonify, current_app, send_file
import os
import subprocess
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create blueprint for report routes
report_bp = Blueprint('report_bp', __name__, url_prefix='/api/generate-report')

@report_bp.route('/', methods=['POST'])
def generate_report():
    """Generate a report for an export."""
    logger.info("Report generation request received")
    
    data = request.json
    logger.info(f"Request data: {data}")
    
    if not data or 'export_name' not in data:
        logger.error("Missing required field: export_name")
        return jsonify({
            'success': False,
            'message': 'Missing required field: export_name'
        }), 400
    
    export_name = data['export_name']
    logger.info(f"Generating report for export: {export_name}")
    
    # Get the correct path to the script
    # Use absolute path resolution based on where this file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_dir = os.path.dirname(current_dir)
    api_dir = os.path.dirname(app_dir)
    project_root = os.path.dirname(api_dir)  # This is the WildlifeDetectionSystem directory
    
    logger.info(f"Project root detected as: {project_root}")
    
    # Path to export_report.py script - check multiple possible locations
    potential_script_paths = [
        os.path.join(project_root, "scripts", "export_report.py"),
        os.path.join(project_root, "api", "scripts", "export_report.py"),
        os.path.join(api_dir, "scripts", "export_report.py")
    ]
    
    script_path = None
    for path in potential_script_paths:
        if os.path.exists(path):
            script_path = path
            logger.info(f"Found script at: {script_path}")
            break
    
    if not script_path:
        error_msg = f"Report generation script not found. Checked paths: {potential_script_paths}"
        logger.error(error_msg)
        return jsonify({
            'success': False,
            'message': error_msg
        }), 500
    
    try:
        # Run the script as a subprocess
        logger.info(f"Running script: python {script_path} {export_name}")
        result = subprocess.run(
            ['python', script_path, export_name], 
            capture_output=True, 
            text=True
        )
        
        if result.returncode == 0:
            logger.info(f"Script executed successfully. Output: {result.stdout}")
            
            # Extract report paths from output
            text_report_path = None
            pdf_report_path = None
            
            for line in result.stdout.split("\n"):
                if "Text report" in line and ":" in line:
                    text_report_path = line.split(":", 1)[1].strip()
                if "PDF report" in line and ":" in line:
                    pdf_report_path = line.split(":", 1)[1].strip()
            
            return jsonify({
                'success': True,
                'message': 'Report generated successfully',
                'text_report_path': text_report_path,
                'pdf_report_path': pdf_report_path
            })
        else:
            error_msg = f"Script execution failed with return code {result.returncode}. Error: {result.stderr}"
            logger.error(error_msg)
            return jsonify({
                'success': False,
                'message': error_msg
            }), 500
    except Exception as e:
        error_msg = f"Error executing script: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            'success': False,
            'message': error_msg
        }), 500