#!/usr/bin/env python3
"""
Seasonal activity analysis tool.
Analyzes species activity patterns across different seasons as per Prof. Peeva's requirements.
"""

import os
import sys
import datetime
import calendar
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Ensure we can import from the app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.models.models import db, Image, Annotation, Species, EnvironmentalData

# Define seasons for Bulgaria
SEASONS = {
    'Winter': [12, 1, 2],
    'Spring': [3, 4, 5],
    'Summer': [6, 7, 8],
    'Autumn': [9, 10, 11]
}

# Define light periods
LIGHT_CONDITIONS = [
    'Full darkness',
    'Early twilight',
    'Late twilight',
    'Daylight'
]

def get_season(month):
    """Get season name from month number."""
    for season, months in SEASONS.items():
        if month in months:
            return season
    return None

def extract_date_info(filename):
    """Extract date information from a filename."""
    # This is a simplified version - extend for more complex filename patterns
    parts = filename.split('_')
    for i, part in enumerate(parts):
        if i < len(parts) - 2 and parts[i].isdigit() and parts[i+1].isdigit():
            try:
                day = int(parts[i])
                month = int(parts[i+1])
                # Check if year follows
                if i+2 < len(parts) and parts[i+2].isdigit():
                    year = int(parts[i+2])
                    if year < 100:
                        year += 2000
                else:
                    year = 2023  # Default year if not specified
                
                if 1 <= day <= 31 and 1 <= month <= 12:
                    return datetime.date(year, month, day)
            except ValueError:
                pass
    return None

def analyze_seasonal_activity():
    """Analyze species activity patterns by season."""
    print("Analyzing seasonal activity patterns...")
    
    # Initialize data structures
    species_seasonal_activity = defaultdict(lambda: defaultdict(int))
    species_light_activity = defaultdict(lambda: defaultdict(int))
    
    # Get all annotations for actual wildlife (not background)
    annotations = (Annotation.query
                  .join(Species)
                  .join(Image)
                  .filter(Species.name != 'Background')
                  .all())
    
    print(f"Found {len(annotations)} non-background annotations to analyze")
    
    for annotation in annotations:
        species_name = annotation.species.name
        image = annotation.image
        date = None
        
        # Try to get timestamp from database first
        if image.timestamp:
            date = image.timestamp.date()
        else:
            # Try to extract from filename
            date = extract_date_info(image.filename)
        
        if date:
            month = date.month
            season = get_season(month)
            if season:
                species_seasonal_activity[species_name][season] += 1
        
        # Get light condition from environmental data if available
        env_data = EnvironmentalData.query.filter_by(image_id=image.id).first()
        if env_data and env_data.light_condition:
            light_condition = env_data.light_condition
            species_light_activity[species_name][light_condition] += 1
    
    # Print results
    print("\nSeasonal Activity Patterns:")
    for species, seasons in species_seasonal_activity.items():
        print(f"\n{species}:")
        total = sum(seasons.values())
        for season, count in seasons.items():
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"  {season}: {count} observations ({percentage:.1f}%)")
    
    print("\nLight Condition Activity Patterns:")
    for species, light_conditions in species_light_activity.items():
        print(f"\n{species}:")
        total = sum(light_conditions.values())
        for condition, count in light_conditions.items():
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"  {condition}: {count} observations ({percentage:.1f}%)")
    
    return species_seasonal_activity, species_light_activity

def generate_charts(species_seasonal_activity, species_light_activity, output_dir="./charts"):
    """Generate visualization charts for the analysis."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process seasonal activity data
    species_names = list(species_seasonal_activity.keys())
    season_names = list(SEASONS.keys())
    
    for species in species_names:
        # Skip if too few data points
        if sum(species_seasonal_activity[species].values()) < 3:
            continue
        
        # Seasonal activity chart
        season_data = [species_seasonal_activity[species].get(season, 0) for season in season_names]
        
        plt.figure(figsize=(10, 6))
        plt.bar(season_names, season_data, color=['lightblue', 'lightgreen', 'gold', 'orange'])
        plt.title(f"Seasonal Activity Pattern: {species}")
        plt.xlabel("Season")
        plt.ylabel("Number of Observations")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        for i, value in enumerate(season_data):
            plt.text(i, value + 0.1, str(value), ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{species}_seasonal_activity.png"))
        plt.close()
        
        # Light condition chart
        if species in species_light_activity:
            light_data = [species_light_activity[species].get(condition, 0) for condition in LIGHT_CONDITIONS]
            
            plt.figure(figsize=(10, 6))
            plt.bar(LIGHT_CONDITIONS, light_data, color=['navy', 'royalblue', 'skyblue', 'yellow'])
            plt.title(f"Activity by Light Condition: {species}")
            plt.xlabel("Light Condition")
            plt.ylabel("Number of Observations")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            for i, value in enumerate(light_data):
                plt.text(i, value + 0.1, str(value), ha='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{species}_light_activity.png"))
            plt.close()
    
    # Generate summary chart
    if len(species_names) > 1:
        plt.figure(figsize=(12, 8))
        
        # Prepare data for grouped bar chart
        bar_width = 0.8 / len(species_names)
        index = np.arange(len(season_names))
        
        for i, species in enumerate(species_names[:5]):  # Limit to 5 species for readability
            season_data = [species_seasonal_activity[species].get(season, 0) for season in season_names]
            position = index + (i * bar_width) - (bar_width * (len(species_names[:5]) - 1) / 2)
            plt.bar(position, season_data, bar_width, label=species)
        
        plt.xlabel('Season')
        plt.ylabel('Number of Observations')
        plt.title('Seasonal Activity by Species')
        plt.xticks(index, season_names)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "species_comparison_seasonal.png"))
        plt.close()
    
    print(f"Charts generated and saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Seasonal activity analysis tool")
    parser.add_argument("--charts", action="store_true", help="Generate visualization charts")
    parser.add_argument("--output", default="./charts", help="Output directory for charts")
    
    args = parser.parse_args()
    
    app = create_app()
    with app.app_context():
        seasonal_data, light_data = analyze_seasonal_activity()
        
        if args.charts:
            generate_charts(seasonal_data, light_data, args.output)

if __name__ == "__main__":
    main()