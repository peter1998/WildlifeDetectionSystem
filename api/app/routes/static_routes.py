from flask import Blueprint, send_from_directory, current_app
from flask import render_template
import os

# Create blueprint for static routes
static_pages = Blueprint('static_pages', __name__)

@static_pages.route('/advanced-annotator')
def advanced_annotator():
    """Serve the advanced annotator HTML page."""
    return render_template('advanced-annotator.html')


@static_pages.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files."""
    static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static')
    return send_from_directory(static_dir, filename)

@static_pages.route('/annotator')
def annotator():
    """Serve the annotator HTML page."""
    static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static')
    return send_from_directory(static_dir, 'annotator.html')

@static_pages.route('/simple-annotator')
def simple_annotator():
    """Serve the simple annotator HTML page."""
    static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static')
    return send_from_directory(static_dir, 'simple-annotator.html')

@static_pages.route('/ground-truth-annotator')
def ground_truth_annotator():
    """Serve the ground truth annotator HTML page."""
    static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static')
    return send_from_directory(static_dir, 'ground-truth-annotator.html')
