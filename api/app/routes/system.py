from flask import Blueprint, jsonify, current_app, request
from app.models.models import Image, Species, Annotation
from app import db
from sqlalchemy import func, distinct
import os
import glob
import json
from datetime import datetime
from app.services.model_performance_service import ModelPerformanceService

# Create a blueprint for system routes
system = Blueprint('system', __name__)

@system.route('/api/system/stats')
def get_system_stats():
    """Get comprehensive system statistics"""
    try:
        # Get statistics directly from database
        stats = {}
        
        # Total Images
        stats['total_images'] = Image.query.count()
        
        # Annotated Images (images with at least one annotation)
        stats['annotated_images'] = db.session.query(
            func.count(distinct(Annotation.image_id))
        ).scalar() or 0
        
        # Species Count
        stats['species_count'] = Species.query.count()
        
        # Image Folders Count
        upload_folder = None
        try:
            # Try to get the upload folder from app config
            upload_folder = current_app.config.get('UPLOAD_FOLDER')
        except:
            # Fallback to common locations if not found in config
            possible_locations = [
                'data/raw_images',
                'raw_images',
                'static/uploads',
                'uploads'
            ]
            
            for loc in possible_locations:
                if os.path.exists(loc) and os.path.isdir(loc):
                    upload_folder = loc
                    break
        
        # Count folders in the upload directory
        if upload_folder and os.path.exists(upload_folder):
            # Count only directories, not files
            stats['folder_count'] = sum(
                1 for item in os.listdir(upload_folder) 
                if os.path.isdir(os.path.join(upload_folder, item))
            )
        else:
            stats['folder_count'] = 0
        
        # Total annotations count
        stats['total_annotations'] = Annotation.query.count()
        
        # Calculate annotation percentage
        if stats['total_images'] > 0:
            stats['annotation_percentage'] = (stats['annotated_images'] / stats['total_images']) * 100
        else:
            stats['annotation_percentage'] = 0
            
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@system.route('/api/system/model-performance')
def get_model_performance():
    """Get comprehensive model performance metrics, optionally for a specific model."""
    try:
        # Check if a specific model_id was requested
        model_id = request.args.get('model_id', None)
        
        if model_id:
            # Get performance for specific model
            model_details = ModelPerformanceService.get_model_details_by_id(model_id)
            performance_metrics = ModelPerformanceService.get_performance_metrics_by_id(model_id)
            confusion_matrix = ModelPerformanceService.get_confusion_matrix_by_id(model_id)
            detection_stats = ModelPerformanceService.get_recent_detection_stats()
            improvement_opportunities = ModelPerformanceService.analyze_improvement_opportunities_by_id(model_id)
            training_history = ModelPerformanceService.get_training_history_by_id(model_id)
        else:
            # Get performance for current/latest model
            model_details = ModelPerformanceService.get_current_model_details()
            performance_metrics = ModelPerformanceService.get_performance_metrics()
            confusion_matrix = ModelPerformanceService.get_confusion_matrix()
            detection_stats = ModelPerformanceService.get_recent_detection_stats()
            improvement_opportunities = ModelPerformanceService.analyze_improvement_opportunities()
            training_history = ModelPerformanceService.get_training_history()
        
        return jsonify({
            'success': True,
            'model_details': model_details,
            'performance_metrics': performance_metrics,
            'confusion_matrix': confusion_matrix,
            'detection_stats': detection_stats,
            'improvement_opportunities': improvement_opportunities,
            'training_history': training_history
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error retrieving model performance: {str(e)}'
        }), 500

@system.route('/api/system/available-models')
def get_available_models():
    """Get list of all available trained models."""
    try:
        # Get trained models directory from config
        models_dir = current_app.config.get('MODEL_FOLDER',
                   os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                   os.path.abspath(__file__)))), 'models', 'trained'))
        
        # Find all model folders
        model_folders = []
        
        if os.path.exists(models_dir):
            for folder in os.listdir(models_dir):
                folder_path = os.path.join(models_dir, folder)
                if os.path.isdir(folder_path) and 'wildlife_detector' in folder:
                    # Get creation time
                    creation_time = os.path.getctime(folder_path)
                    creation_date = datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Check for weights files
                    weights_files = []
                    for weights_file in ['best.pt', 'last.pt']:
                        weights_path = os.path.join(folder_path, 'weights', weights_file)
                        if os.path.exists(weights_path):
                            weights_files.append(weights_file)
                        
                        # Also check directly in model folder
                        weights_path = os.path.join(folder_path, weights_file)
                        if os.path.exists(weights_path) and weights_file not in weights_files:
                            weights_files.append(weights_file)
                    
                    # Get model type from metadata if available
                    model_type = "YOLOv8"
                    metadata_path = os.path.join(folder_path, 'model_metadata.json')
                    if os.path.exists(metadata_path):
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                                if 'model_type' in metadata:
                                    model_type = metadata['model_type']
                        except Exception:
                            pass
                    
                    # Add to list
                    model_folders.append({
                        'id': folder,
                        'name': folder,
                        'created_at': creation_date,
                        'creation_time': creation_time,
                        'weights_files': weights_files,
                        'model_type': model_type
                    })
        
        # Sort by creation time (newest first)
        model_folders.sort(key=lambda x: x['creation_time'], reverse=True)
        
        # Remove creation_time from response as it's just for sorting
        for model in model_folders:
            model.pop('creation_time', None)
        
        return jsonify({
            'success': True,
            'models': model_folders
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error retrieving available models: {str(e)}'
        }), 500

@system.route('/api/system/model-comparison')
def compare_models():
    """Compare multiple models by ID."""
    try:
        # Get model IDs from request
        model_ids = request.args.getlist('model_ids[]')
        
        if not model_ids:
            return jsonify({
                'success': False,
                'message': 'No model IDs provided'
            }), 400
        
        # Get performance data for each model
        models_data = []
        
        for model_id in model_ids:
            model_details = ModelPerformanceService.get_model_details_by_id(model_id)
            performance_metrics = ModelPerformanceService.get_performance_metrics_by_id(model_id)
            
            if model_details and performance_metrics:
                models_data.append({
                    'id': model_id,
                    'details': model_details,
                    'metrics': performance_metrics
                })
        
        return jsonify({
            'success': True,
            'models': models_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error comparing models: {str(e)}'
        }), 500

@system.route('/api/system/threshold-analysis/<model_id>')
def get_threshold_analysis(model_id):
    """Get performance metrics across different confidence thresholds for a model."""
    try:
        # Get threshold analysis from results directory if available
        models_dir = current_app.config.get('MODEL_FOLDER',
                   os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                   os.path.abspath(__file__)))), 'models', 'trained'))
        
        model_dir = os.path.join(models_dir, model_id)
        threshold_path = os.path.join(model_dir, 'threshold_analysis.json')
        
        if os.path.exists(threshold_path):
            with open(threshold_path, 'r') as f:
                threshold_data = json.load(f)
                
            return jsonify({
                'success': True,
                'thresholds': threshold_data
            })
        else:
            # Try to find in reports directory
            reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                        os.path.abspath(__file__)))), 'reports')
            
            # Look for evaluation reports that might contain threshold analysis
            evaluation_dirs = glob.glob(os.path.join(reports_dir, 'evaluation_*'))
            threshold_data = None
            
            for eval_dir in sorted(evaluation_dirs, reverse=True):  # Newest first
                threshold_path = os.path.join(eval_dir, 'threshold_analysis.json')
                if os.path.exists(threshold_path):
                    with open(threshold_path, 'r') as f:
                        threshold_data = json.load(f)
                    break
            
            if threshold_data:
                return jsonify({
                    'success': True,
                    'thresholds': threshold_data
                })
            else:
                # Return simulated data for demonstration
                thresholds = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90]
                threshold_data = []
                
                # Default values that decay/improve with threshold changes
                base_precision = 0.75
                base_recall = 0.50
                base_map50 = 0.60
                
                for i, threshold in enumerate(thresholds):
                    # Higher threshold = higher precision, lower recall
                    precision_factor = 1.0 + (threshold - 0.5) * 0.5
                    recall_factor = 1.0 - (threshold - 0.25) * 1.2
                    
                    precision = min(0.95, max(0.3, base_precision * precision_factor))
                    recall = min(0.9, max(0.1, base_recall * recall_factor))
                    map50 = min(0.85, max(0.3, base_map50 * (precision + recall) / (base_precision + base_recall)))
                    
                    threshold_data.append({
                        'threshold': threshold,
                        'precision': precision,
                        'recall': recall,
                        'mAP50': map50
                    })
                
                return jsonify({
                    'success': True,
                    'thresholds': threshold_data,
                    'simulated': True
                })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error retrieving threshold analysis: {str(e)}'
        }), 500

@system.route('/api/system/taxonomic-performance/<model_id>')
def get_taxonomic_performance(model_id):
    """Get performance metrics broken down by taxonomic group."""
    try:
        # Define taxonomic groups
        taxonomic_groups = {
            "Deer": ["Red Deer", "Male Roe Deer", "Female Roe Deer", "Fallow Deer"],
            "Carnivores": ["Fox", "Wolf", "Jackal", "Brown Bear", "Badger", "Weasel", 
                          "Stoat", "Polecat", "Marten", "Otter", "Wildcat"],
            "Small_Mammals": ["Rabbit", "Hare", "Squirrel", "Dormouse", "Hedgehog"],
            "Birds": ["Blackbird", "Nightingale", "Pheasant", "woodpecker"],
            "Other": ["Wild Boar", "Chamois", "Turtle", "Human", "Background", "Dog"]
        }
        
        # Get performance metrics for specified model
        if model_id == 'current':
            performance_metrics = ModelPerformanceService.get_performance_metrics()
        else:
            performance_metrics = ModelPerformanceService.get_performance_metrics_by_id(model_id)
        
        if not performance_metrics or 'per_class' not in performance_metrics:
            return jsonify({
                'success': False,
                'message': 'No per-class metrics available for taxonomic analysis'
            }), 404
        
        # Calculate metrics for each taxonomic group
        group_metrics = {}
        
        # Initialize groups
        for group in taxonomic_groups:
            group_metrics[group] = {
                'precision': 0,
                'recall': 0,
                'map50': 0,
                'count': 0,
                'species': []
            }
        
        # Accumulate metrics for each group
        for species, stats in performance_metrics['per_class'].items():
            # Find which group this species belongs to
            for group, species_list in taxonomic_groups.items():
                if species in species_list:
                    group_metrics[group]['precision'] += stats.get('precision', 0)
                    group_metrics[group]['recall'] += stats.get('recall', 0)
                    group_metrics[group]['map50'] += stats.get('map50', 0)
                    group_metrics[group]['count'] += 1
                    group_metrics[group]['species'].append({
                        'name': species,
                        'precision': stats.get('precision', 0),
                        'recall': stats.get('recall', 0),
                        'map50': stats.get('map50', 0)
                    })
                    break
        
        # Calculate averages
        for group in group_metrics:
            if group_metrics[group]['count'] > 0:
                group_metrics[group]['precision'] /= group_metrics[group]['count']
                group_metrics[group]['recall'] /= group_metrics[group]['count']
                group_metrics[group]['map50'] /= group_metrics[group]['count']
        
        return jsonify({
            'success': True,
            'taxonomic_groups': group_metrics
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error retrieving taxonomic performance: {str(e)}'
        }), 500