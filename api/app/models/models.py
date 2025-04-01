from datetime import datetime
from app import db

class Image(db.Model):
    """
    Model representing an uploaded camera trap image.
    """
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False, unique=True)
    original_path = db.Column(db.String(512), nullable=False)
    processed_path = db.Column(db.String(512), nullable=True)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    width = db.Column(db.Integer, nullable=True)
    height = db.Column(db.Integer, nullable=True)
    
    # Metadata
    location = db.Column(db.String(255), nullable=True)
    timestamp = db.Column(db.DateTime, nullable=True)
    camera_id = db.Column(db.String(100), nullable=True)
    
    # Relationships
    annotations = db.relationship('Annotation', backref='image', lazy=True, cascade="all, delete-orphan")
    environmental_data = db.relationship('EnvironmentalData', backref='image', lazy=True, cascade="all, delete-orphan", uselist=False)
    
    def __repr__(self):
        return f'<Image {self.filename}>'

class Species(db.Model):
    """
    Model representing wildlife species for classification.
    """
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    scientific_name = db.Column(db.String(255), nullable=True)
    description = db.Column(db.Text, nullable=True)
    
    # Relationships
    annotations = db.relationship('Annotation', backref='species', lazy=True)
    sequence_events = db.relationship('SequenceEvent', backref='species', lazy=True)
    
    def __repr__(self):
        return f'<Species {self.name}>'

class Annotation(db.Model):
    """
    Model representing a bounding box annotation for an image.
    """
    id = db.Column(db.Integer, primary_key=True)
    image_id = db.Column(db.Integer, db.ForeignKey('image.id'), nullable=False)
    species_id = db.Column(db.Integer, db.ForeignKey('species.id'), nullable=False)
    
    # Bounding box coordinates (normalized 0-1)
    x_min = db.Column(db.Float, nullable=False)
    y_min = db.Column(db.Float, nullable=False)
    x_max = db.Column(db.Float, nullable=False)
    y_max = db.Column(db.Float, nullable=False)
    
    # Metadata
    confidence = db.Column(db.Float, nullable=True)  # For model predictions
    is_verified = db.Column(db.Boolean, default=False)  # Human verification flag
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    behavioral_notes = db.relationship('BehavioralNote', backref='annotation', lazy=True, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f'<Annotation {self.id} for Image {self.image_id}>'

class EnvironmentalData(db.Model):
    """
    Environmental data for diurnal activity analysis as per Prof. Peeva's requirements.
    """
    id = db.Column(db.Integer, primary_key=True)
    image_id = db.Column(db.Integer, db.ForeignKey('image.id'), nullable=False, unique=True)
    
    # Light conditions
    light_condition = db.Column(db.String(50), nullable=True)  # Full darkness, Early twilight, Late twilight, Daylight
    
    # Environmental factors
    temperature = db.Column(db.Float, nullable=True)  # Temperature in Celsius
    moon_phase = db.Column(db.String(50), nullable=True)  # Moon phase (Full, New, etc.)
    snow_cover = db.Column(db.Boolean, nullable=True)  # Snow cover present
    
    # Habitat data
    vegetation_type = db.Column(db.String(100), nullable=True)  # Type of vegetation
    habitat_type = db.Column(db.String(100), nullable=True)  # Plains or mountains (Stara Planina)
    
    def __repr__(self):
        return f'<EnvironmentalData for Image {self.image_id}>'

class BehavioralNote(db.Model):
    """
    Notes about animal behavior for behavioral tracking.
    """
    id = db.Column(db.Integer, primary_key=True)
    annotation_id = db.Column(db.Integer, db.ForeignKey('annotation.id'), nullable=False)
    
    behavior_type = db.Column(db.String(100), nullable=False)  # Behavior category
    notes = db.Column(db.Text, nullable=True)  # Detailed notes
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<BehavioralNote {self.id} for Annotation {self.annotation_id}>'

class SequenceEvent(db.Model):
    """
    Tracks chronological appearances of animals for predator-prey analysis.
    """
    id = db.Column(db.Integer, primary_key=True)
    location = db.Column(db.String(255), nullable=False)  # Camera location
    species_id = db.Column(db.Integer, db.ForeignKey('species.id'), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)  # When animal was detected
    
    # Sequence tracking
    previous_event_id = db.Column(db.Integer, db.ForeignKey('sequence_event.id'), nullable=True)
    time_since_previous = db.Column(db.Integer, nullable=True)  # Time in seconds
    
    # Self-referential relationship
    next_events = db.relationship('SequenceEvent', 
                                  backref=db.backref('previous_event', remote_side=[id]),
                                  foreign_keys=[previous_event_id])
    
    def __repr__(self):
        return f'<SequenceEvent {self.id} for Species {self.species_id}>'
