#!/usr/bin/env python3
"""
Script to update environmental data for existing annotations.
This adds light conditions and other environmental factors based on image metadata.
"""

import os
import sys
import datetime
import re

# Ensure we can import from the app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.models.models import db, Image, Annotation, EnvironmentalData, Species
from app.services.environmental_service import EnvironmentalService

def extract_date_from_filename(filename):
    """Extract date information from filename if available."""
    # Common date patterns in the filenames
    date_patterns = [
        r'(\d{1,2})_(\d{1,2})__(\d{4})',  # DD_MM__YYYY
        r'(\d{1,2})_(\d{1,2})_(\d{2,4})',  # DD_MM_YY or DD_MM_YYYY
        r'(\d{2})_(\d{2})_(\d{2,4})'       # DD_MM_YY or DD_MM_YYYY
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, filename)
        if match:
            day, month, year = match.groups()
            if len(year) == 2:
                year = f"20{year}"  # Assume 20xx for 2-digit years
            try:
                return datetime.datetime(int(year), int(month), int(day))
            except ValueError:
                pass
    
    return None

def determine_light_condition(filename):
    """
    Determine light condition based on filename patterns.
    This is a basic heuristic that could be replaced with actual image analysis.
    """
    # Check for night indicators
    night_indicators = ['night', 'darkness', 'dark', 'evening']
    twilight_indicators = ['twilight', 'dawn', 'dusk', 'sunrise', 'sunset']
    
    filename_lower = filename.lower()
    
    for indicator in night_indicators:
        if indicator in filename_lower:
            return "Full darkness"
    
    for indicator in twilight_indicators:
        if indicator in filename_lower:
            if 'early' in filename_lower or 'dawn' in filename_lower or 'sunrise' in filename_lower:
                return "Early twilight"
            elif 'late' in filename_lower or 'dusk' in filename_lower or 'sunset' in filename_lower:
                return "Late twilight"
            return "Twilight"
    
    # Default assumption based on image clarity - most camera trap images are during daylight
    # This should be replaced with actual image analysis
    return "Daylight"

def determine_habitat_type(filename):
    """Determine habitat type based on location indicators in filename."""
    filename_lower = filename.lower()
    
    if 'mountain' in filename_lower or 'planina' in filename_lower or 'stara' in filename_lower:
        return "Mountains"
    elif 'plain' in filename_lower or 'field' in filename_lower or 'meadow' in filename_lower:
        return "Plains"
    
    return "Unknown"

def determine_snow_cover(filename, timestamp=None):
    """Determine if snow cover is likely based on month."""
    if timestamp:
        # Winter months in Bulgaria often have snow
        if timestamp.month in [12, 1, 2, 3]:
            return True
    
    # Check filename for snow indicators
    snow_indicators = ['snow', 'winter', 'snowcover', 'сняг']
    filename_lower = filename.lower()
    
    for indicator in snow_indicators:
        if indicator in filename_lower:
            return True
    
    return False

def determine_vegetation_type(filename):
    """Determine vegetation type based on filename and location."""
    filename_lower = filename.lower()
    
    if 'forest' in filename_lower or 'woods' in filename_lower or 'tree' in filename_lower:
        return "Forest"
    elif 'meadow' in filename_lower or 'grass' in filename_lower or 'field' in filename_lower:
        return "Meadow"
    elif 'bush' in filename_lower or 'shrub' in filename_lower:
        return "Shrubland"
    
    return "Mixed"

def update_environmental_data():
    """Update environmental data for all images in the database."""
    print("Starting environmental data update...")
    
    # Get all images
    images = Image.query.all()
    print(f"Found {len(images)} images to process")
    
    updated_count = 0
    created_count = 0
    
    for image in images:
        print(f"Processing image: {image.filename}")
        
        # Check if environmental data already exists
        env_data = EnvironmentalData.query.filter_by(image_id=image.id).first()
        
        # Extract date from filename if timestamp is not set
        timestamp = image.timestamp
        if not timestamp:
            timestamp = extract_date_from_filename(image.filename)
        
        # Determine environmental conditions
        light_condition = determine_light_condition(image.filename)
        habitat_type = determine_habitat_type(image.filename)
        vegetation_type = determine_vegetation_type(image.filename)
        snow_cover = determine_snow_cover(image.filename, timestamp)
        
        # Set a default temperature range based on month if timestamp available
        temperature = None
        if timestamp:
            # Very basic temperature estimation by month for Bulgaria
            month_temp_map = {
                1: 0,    # January
                2: 2,    # February
                3: 8,    # March
                4: 13,   # April
                5: 18,   # May
                6: 22,   # June
                7: 25,   # July
                8: 25,   # August
                9: 20,   # September
                10: 15,  # October
                11: 8,   # November
                12: 2    # December
            }
            temperature = month_temp_map.get(timestamp.month)
        
        try:
            if env_data:
                # Update existing record
                env_data.light_condition = light_condition
                env_data.habitat_type = habitat_type
                env_data.vegetation_type = vegetation_type
                env_data.snow_cover = snow_cover
                if temperature:
                    env_data.temperature = temperature
                db.session.commit()
                updated_count += 1
            else:
                # Create new record
                env_data = EnvironmentalData(
                    image_id=image.id,
                    light_condition=light_condition,
                    habitat_type=habitat_type,
                    vegetation_type=vegetation_type,
                    snow_cover=snow_cover,
                    temperature=temperature
                )
                db.session.add(env_data)
                db.session.commit()
                created_count += 1
            
            print(f"  Updated environmental data: {light_condition}, {habitat_type}, {vegetation_type}")
        
        except Exception as e:
            print(f"  Error updating environmental data: {str(e)}")
            db.session.rollback()
    
    print(f"Completed updating environmental data.")
    print(f"Created: {created_count}, Updated: {updated_count}")

if __name__ == "__main__":
    app = create_app()
    with app.app_context():
        update_environmental_data()