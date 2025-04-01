from app import db
from app.models.models import Image, EnvironmentalData, BehavioralNote, Annotation, SequenceEvent, Species
from datetime import datetime

class EnvironmentalService:
    """Service for managing environmental data."""
    
    @staticmethod
    def get_environmental_data(image_id):
        """Get environmental data for an image."""
        data = EnvironmentalData.query.filter_by(image_id=image_id).first()
        return data
    
    @staticmethod
    def create_or_update_environmental_data(image_id, light_condition=None, temperature=None, 
                                           moon_phase=None, snow_cover=None, vegetation_type=None,
                                           habitat_type=None):
        """Create or update environmental data for an image."""
        # Check if image exists
        image = Image.query.get_or_404(image_id)
        
        # Check if environmental data already exists
        env_data = EnvironmentalData.query.filter_by(image_id=image_id).first()
        if not env_data:
            env_data = EnvironmentalData(image_id=image_id)
            db.session.add(env_data)
        
        # Update fields if provided
        if light_condition is not None:
            env_data.light_condition = light_condition
        if temperature is not None:
            env_data.temperature = temperature
        if moon_phase is not None:
            env_data.moon_phase = moon_phase
        if snow_cover is not None:
            env_data.snow_cover = snow_cover
        if vegetation_type is not None:
            env_data.vegetation_type = vegetation_type
        if habitat_type is not None:
            env_data.habitat_type = habitat_type
        
        try:
            db.session.commit()
            return env_data
        except Exception as e:
            db.session.rollback()
            raise e
    
    @staticmethod
    def delete_environmental_data(image_id):
        """Delete environmental data for an image."""
        env_data = EnvironmentalData.query.filter_by(image_id=image_id).first()
        if env_data:
            db.session.delete(env_data)
            db.session.commit()
            return True
        return False
    
    @staticmethod
    def analyze_light_conditions(image_path):
        """
        Automatically analyze an image to determine light conditions.
        This is a placeholder for actual implementation that would use
        computer vision techniques to analyze image brightness.
        """
        # TODO: Implement actual analysis using OpenCV
        # For now, return a placeholder result
        return "Unknown"


class BehavioralTrackingService:
    """Service for tracking animal behaviors and sequences."""
    
    @staticmethod
    def add_behavioral_note(annotation_id, behavior_type, notes=None):
        """Add a behavioral note to an annotation."""
        # Check if annotation exists
        annotation = Annotation.query.get_or_404(annotation_id)
        
        note = BehavioralNote(
            annotation_id=annotation_id,
            behavior_type=behavior_type,
            notes=notes
        )
        
        try:
            db.session.add(note)
            db.session.commit()
            return note
        except Exception as e:
            db.session.rollback()
            raise e
    
    @staticmethod
    def get_behavioral_notes(annotation_id):
        """Get all behavioral notes for an annotation."""
        notes = BehavioralNote.query.filter_by(annotation_id=annotation_id).all()
        return notes
    
    @staticmethod
    def delete_behavioral_note(note_id):
        """Delete a behavioral note."""
        note = BehavioralNote.query.get_or_404(note_id)
        
        try:
            db.session.delete(note)
            db.session.commit()
            return True
        except Exception as e:
            db.session.rollback()
            raise e
    
    @staticmethod
    def record_sequence_event(location, species_id, timestamp, previous_event_id=None):
        """Record a sequence event for chronological tracking."""
        # Check if species exists
        species = Species.query.get_or_404(species_id)
        
        # Calculate time since previous if provided
        time_since_previous = None
        if previous_event_id:
            previous_event = SequenceEvent.query.get_or_404(previous_event_id)
            if previous_event.timestamp and timestamp:
                time_diff = timestamp - previous_event.timestamp
                time_since_previous = time_diff.total_seconds()
        
        # Create event
        event = SequenceEvent(
            location=location,
            species_id=species_id,
            timestamp=timestamp,
            previous_event_id=previous_event_id,
            time_since_previous=time_since_previous
        )
        
        try:
            db.session.add(event)
            db.session.commit()
            return event
        except Exception as e:
            db.session.rollback()
            raise e
    
    @staticmethod
    def get_sequence_events(location=None, species_id=None, start_date=None, end_date=None):
        """Get sequence events filtered by criteria."""
        query = SequenceEvent.query
        
        if location:
            query = query.filter_by(location=location)
        
        if species_id:
            query = query.filter_by(species_id=species_id)
        
        if start_date:
            query = query.filter(SequenceEvent.timestamp >= start_date)
        
        if end_date:
            query = query.filter(SequenceEvent.timestamp <= end_date)
        
        return query.order_by(SequenceEvent.timestamp).all()
    
    @staticmethod
    def analyze_predator_prey_patterns(location, start_date=None, end_date=None):
        """
        Analyze predator-prey patterns based on sequence events.
        
        Returns:
            dict: Summary of predator-prey patterns
        """
        # Get all events for the location in chronological order
        events = BehavioralTrackingService.get_sequence_events(
            location=location,
            start_date=start_date,
            end_date=end_date
        )
        
        # For simplicity, we'll define some predator and prey species
        # In a real implementation, this would be configurable
        predator_species = ['Wolf', 'Fox', 'Jackal', 'Brown Bear']
        prey_species = ['Red Deer', 'Male Roe Deer', 'Female Roe Deer', 'Rabbit', 'Hare']
        
        # Analyze event sequences
        patterns = {
            'predator_followed_by_prey': 0,
            'prey_followed_by_predator': 0,
            'average_time_between': None,
            'total_time_measured': 0,
            'event_count': len(events)
        }
        
        total_time = 0
        sequence_count = 0
        
        # Analyze consecutive events
        for i in range(1, len(events)):
            current = events[i]
            previous = events[i-1]
            
            current_species = Species.query.get(current.species_id)
            previous_species = Species.query.get(previous.species_id)
            
            if not current_species or not previous_species:
                continue
                
            # Count predator-prey sequences
            if previous_species.name in predator_species and current_species.name in prey_species:
                patterns['predator_followed_by_prey'] += 1
                
            if previous_species.name in prey_species and current_species.name in predator_species:
                patterns['prey_followed_by_predator'] += 1
            
            # Calculate time between events
            if current.time_since_previous:
                total_time += current.time_since_previous
                sequence_count += 1
        
        # Calculate average time
        if sequence_count > 0:
            patterns['average_time_between'] = total_time / sequence_count
            patterns['total_time_measured'] = total_time
        
        return patterns