from flask import Blueprint, request, jsonify
from app import db
from app.models.models import EnvironmentalData, BehavioralNote, SequenceEvent
from app.services.environmental_service import EnvironmentalService, BehavioralTrackingService
from datetime import datetime

# Create blueprint for environmental data routes
environmental = Blueprint('environmental', __name__, url_prefix='/api/environmental')

@environmental.route('/image/<int:image_id>', methods=['GET'])
def get_environmental_data(image_id):
    """Get environmental data for a specific image."""
    try:
        env_data = EnvironmentalService.get_environmental_data(image_id)
        
        if not env_data:
            return jsonify({
                'success': True,
                'message': 'No environmental data found for this image',
                'data': None
            }), 200
        
        result = {
            'success': True,
            'data': {
                'id': env_data.id,
                'image_id': env_data.image_id,
                'light_condition': env_data.light_condition,
                'temperature': env_data.temperature,
                'moon_phase': env_data.moon_phase,
                'snow_cover': env_data.snow_cover,
                'vegetation_type': env_data.vegetation_type,
                'habitat_type': env_data.habitat_type
            }
        }
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@environmental.route('/image/<int:image_id>', methods=['POST', 'PUT'])
def create_or_update_environmental_data(image_id):
    """Create or update environmental data for a specific image."""
    data = request.json
    
    if not data:
        return jsonify({
            'success': False,
            'message': 'No data provided'
        }), 400
    
    try:
        env_data = EnvironmentalService.create_or_update_environmental_data(
            image_id=image_id,
            light_condition=data.get('light_condition'),
            temperature=data.get('temperature'),
            moon_phase=data.get('moon_phase'),
            snow_cover=data.get('snow_cover'),
            vegetation_type=data.get('vegetation_type'),
            habitat_type=data.get('habitat_type')
        )
        
        result = {
            'success': True,
            'message': 'Environmental data updated successfully',
            'data': {
                'id': env_data.id,
                'image_id': env_data.image_id,
                'light_condition': env_data.light_condition,
                'temperature': env_data.temperature,
                'moon_phase': env_data.moon_phase,
                'snow_cover': env_data.snow_cover,
                'vegetation_type': env_data.vegetation_type,
                'habitat_type': env_data.habitat_type
            }
        }
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@environmental.route('/image/<int:image_id>', methods=['DELETE'])
def delete_environmental_data(image_id):
    """Delete environmental data for a specific image."""
    try:
        success = EnvironmentalService.delete_environmental_data(image_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Environmental data deleted successfully'
            }), 200
        else:
            return jsonify({
                'success': True,
                'message': 'No environmental data found for this image'
            }), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

# Routes for behavioral tracking
@environmental.route('/behavior/annotation/<int:annotation_id>', methods=['POST'])
def add_behavioral_note(annotation_id):
    """Add a behavioral note to an annotation."""
    data = request.json
    
    if not data or 'behavior_type' not in data:
        return jsonify({
            'success': False,
            'message': 'Behavior type is required'
        }), 400
    
    try:
        note = BehavioralTrackingService.add_behavioral_note(
            annotation_id=annotation_id,
            behavior_type=data['behavior_type'],
            notes=data.get('notes')
        )
        
        result = {
            'success': True,
            'message': 'Behavioral note added successfully',
            'note': {
                'id': note.id,
                'annotation_id': note.annotation_id,
                'behavior_type': note.behavior_type,
                'notes': note.notes,
                'created_at': note.created_at.isoformat()
            }
        }
        
        return jsonify(result), 201
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@environmental.route('/behavior/annotation/<int:annotation_id>', methods=['GET'])
def get_behavioral_notes(annotation_id):
    """Get all behavioral notes for an annotation."""
    try:
        notes = BehavioralTrackingService.get_behavioral_notes(annotation_id)
        
        result = {
            'success': True,
            'notes': [{
                'id': note.id,
                'annotation_id': note.annotation_id,
                'behavior_type': note.behavior_type,
                'notes': note.notes,
                'created_at': note.created_at.isoformat()
            } for note in notes]
        }
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@environmental.route('/behavior/note/<int:note_id>', methods=['DELETE'])
def delete_behavioral_note(note_id):
    """Delete a behavioral note."""
    try:
        success = BehavioralTrackingService.delete_behavioral_note(note_id)
        
        return jsonify({
            'success': success,
            'message': 'Behavioral note deleted successfully'
        }), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

# Routes for sequence tracking
@environmental.route('/sequence', methods=['POST'])
def record_sequence_event():
    """Record a new sequence event."""
    data = request.json
    
    if not data or 'location' not in data or 'species_id' not in data or 'timestamp' not in data:
        return jsonify({
            'success': False,
            'message': 'Location, species_id, and timestamp are required'
        }), 400
    
    try:
        # Parse timestamp from ISO format
        timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
        
        event = BehavioralTrackingService.record_sequence_event(
            location=data['location'],
            species_id=data['species_id'],
            timestamp=timestamp,
            previous_event_id=data.get('previous_event_id')
        )
        
        result = {
            'success': True,
            'message': 'Sequence event recorded successfully',
            'event': {
                'id': event.id,
                'location': event.location,
                'species_id': event.species_id,
                'timestamp': event.timestamp.isoformat(),
                'previous_event_id': event.previous_event_id,
                'time_since_previous': event.time_since_previous
            }
        }
        
        return jsonify(result), 201
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@environmental.route('/sequence', methods=['GET'])
def get_sequence_events():
    """Get sequence events filtered by criteria."""
    location = request.args.get('location')
    species_id = request.args.get('species_id', type=int)
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    # Parse dates if provided
    if start_date:
        start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
    if end_date:
        end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
    
    try:
        events = BehavioralTrackingService.get_sequence_events(
            location=location,
            species_id=species_id,
            start_date=start_date,
            end_date=end_date
        )
        
        result = {
            'success': True,
            'events': [{
                'id': event.id,
                'location': event.location,
                'species_id': event.species_id,
                'timestamp': event.timestamp.isoformat(),
                'previous_event_id': event.previous_event_id,
                'time_since_previous': event.time_since_previous
            } for event in events]
        }
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@environmental.route('/sequence/analyze/<location>', methods=['GET'])
def analyze_predator_prey_patterns(location):
    """Analyze predator-prey patterns for a location."""
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    # Parse dates if provided
    if start_date:
        start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
    if end_date:
        end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
    
    try:
        patterns = BehavioralTrackingService.analyze_predator_prey_patterns(
            location=location,
            start_date=start_date,
            end_date=end_date
        )
        
        result = {
            'success': True,
            'location': location,
            'patterns': patterns
        }
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500