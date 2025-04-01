from flask import Blueprint, request, jsonify
from app.services.species_service import SpeciesService

# Create blueprint for species routes
species = Blueprint('species', __name__, url_prefix='/api/species')

@species.route('/', methods=['GET'])
def get_all_species():
    """Get all species."""
    try:
        all_species = SpeciesService.get_all_species()
        
        result = {
            'success': True,
            'species': [{
                'id': s.id,
                'name': s.name,
                'scientific_name': s.scientific_name,
                'description': s.description
            } for s in all_species]
        }
        
        return jsonify(result), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error retrieving species: {str(e)}'
        }), 500

@species.route('/<int:species_id>', methods=['GET'])
def get_species(species_id):
    """Get a specific species by ID."""
    species = SpeciesService.get_species_by_id(species_id)
    
    if not species:
        return jsonify({
            'success': False,
            'message': 'Species not found'
        }), 404
    
    result = {
        'success': True,
        'species': {
            'id': species.id,
            'name': species.name,
            'scientific_name': species.scientific_name,
            'description': species.description
        }
    }
    
    return jsonify(result), 200

@species.route('/', methods=['POST'])
def create_species():
    """Create a new species."""
    data = request.get_json()
    
    if not data or 'name' not in data:
        return jsonify({
            'success': False,
            'message': 'Name is required'
        }), 400
    
    try:
        species = SpeciesService.create_species(
            name=data['name'],
            scientific_name=data.get('scientific_name'),
            description=data.get('description')
        )
        
        result = {
            'success': True,
            'message': 'Species created successfully',
            'species': {
                'id': species.id,
                'name': species.name,
                'scientific_name': species.scientific_name,
                'description': species.description
            }
        }
        
        return jsonify(result), 201
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error creating species: {str(e)}'
        }), 500

@species.route('/<int:species_id>', methods=['PUT'])
def update_species(species_id):
    """Update an existing species."""
    data = request.get_json()
    
    if not data:
        return jsonify({
            'success': False,
            'message': 'No data provided'
        }), 400
    
    try:
        species = SpeciesService.update_species(
            species_id=species_id,
            name=data.get('name'),
            scientific_name=data.get('scientific_name'),
            description=data.get('description')
        )
        
        if not species:
            return jsonify({
                'success': False,
                'message': 'Species not found'
            }), 404
        
        result = {
            'success': True,
            'message': 'Species updated successfully',
            'species': {
                'id': species.id,
                'name': species.name,
                'scientific_name': species.scientific_name,
                'description': species.description
            }
        }
        
        return jsonify(result), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error updating species: {str(e)}'
        }), 500

@species.route('/<int:species_id>', methods=['DELETE'])
def delete_species(species_id):
    """Delete a species."""
    try:
        success = SpeciesService.delete_species(species_id)
        
        if not success:
            return jsonify({
                'success': False,
                'message': 'Species not found'
            }), 404
        
        return jsonify({
            'success': True,
            'message': 'Species deleted successfully'
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error deleting species: {str(e)}'
        }), 500

@species.route('/initialize', methods=['POST'])
def initialize_default_species():
    """Initialize the database with default species."""
    try:
        count = SpeciesService.initialize_default_species()
        
        return jsonify({
            'success': True,
            'message': f'Successfully initialized {count} default species',
            'count': count
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error initializing species: {str(e)}'
        }), 500
