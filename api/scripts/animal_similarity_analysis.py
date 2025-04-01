#!/usr/bin/env python3
"""
Animal Similarity Analysis Tool

This script helps analyze similarities between species like wolf and jackal,
as mentioned in Prof. Peeva's requirements. It extracts features from existing
annotations to help train models to distinguish similar species.
"""

import os
import sys
import argparse
import numpy as np
import cv2
from PIL import Image as PILImage
from pathlib import Path

# Ensure we can import from the app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.models.models import db, Image, Annotation, Species
from app.services.image_service import ImageService

# Helper functions
def extract_box(image_path, x_min, y_min, x_max, y_max):
    """Extract a bounding box from an image."""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image {image_path}")
            return None
            
        height, width = img.shape[:2]
        
        # Convert normalized coordinates to pixel coordinates
        x_min_px = int(x_min * width)
        y_min_px = int(y_min * height)
        x_max_px = int(x_max * width)
        y_max_px = int(y_max * height)
        
        # Extract the box
        box = img[y_min_px:y_max_px, x_min_px:x_max_px]
        
        return box
    except Exception as e:
        print(f"Error extracting box: {str(e)}")
        return None

def calculate_histogram(image):
    """Calculate color histogram for an image."""
    try:
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate histogram
        hist_h = cv2.calcHist([hsv], [0], None, [30], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        
        # Normalize
        cv2.normalize(hist_h, hist_h, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_s, hist_s, 0, 1, cv2.NORM_MINMAX)
        
        # Combine
        return np.concatenate([hist_h, hist_s]).flatten()
    except Exception as e:
        print(f"Error calculating histogram: {str(e)}")
        return None

def extract_shape_features(image):
    """Extract shape features using contours."""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If no contours found, return zeros
        if not contours:
            return np.zeros(6)
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate features
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        extent = area / (w * h) if w * h > 0 else 0
        
        # Calculate moments
        M = cv2.moments(largest_contour)
        hu_moments = cv2.HuMoments(M).flatten()
        
        # Combine basic shape features
        basic_features = np.array([area, perimeter, aspect_ratio, extent])
        
        # Return first 2 Hu moments (most important) combined with basic features
        return np.concatenate([basic_features, hu_moments[:2]])
    except Exception as e:
        print(f"Error extracting shape features: {str(e)}")
        return np.zeros(6)

def analyze_species_similarities(species_names=None, output_dir="./similarity_analysis"):
    """
    Analyze similarities between specified species using their annotations.
    If no species names are provided, focus on wolf/jackal differentiation.
    """
    print("Analyzing species similarities...")
    
    # Default to wolf/jackal if no species specified
    if not species_names:
        species_names = ['Wolf', 'Jackal']
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get species IDs
    species_map = {}
    for name in species_names:
        species = Species.query.filter_by(name=name).first()
        if species:
            species_map[species.id] = species.name
    
    if not species_map:
        print(f"Error: None of the specified species found in database")
        return
    
    print(f"Analyzing similarities between: {', '.join(species_map.values())}")
    
    # Get annotations for the specified species
    annotations = (Annotation.query
                  .join(Species)
                  .join(Image)
                  .filter(Species.id.in_(species_map.keys()))
                  .all())
    
    print(f"Found {len(annotations)} annotations for analysis")
    
    # Process annotations
    results = []
    extraction_failures = 0
    
    for annotation in annotations:
        image = annotation.image
        species_name = species_map[annotation.species_id]
        
        # Construct full image path
        data_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).parent / "data"
        image_path = data_dir / image.original_path.replace("./", "")
        
        print(f"Processing {image.filename} - {species_name}...")
        
        # Extract box
        box = extract_box(
            str(image_path),
            annotation.x_min,
            annotation.y_min,
            annotation.x_max,
            annotation.y_max
        )
        
        if box is None or box.size == 0:
            print(f"  Error: Could not extract box for {image.filename}")
            extraction_failures += 1
            continue
        
        # Calculate features
        try:
            # Resize for consistency
            box_resized = cv2.resize(box, (100, 100))
            
            # Extract features
            color_histogram = calculate_histogram(box_resized)
            shape_features = extract_shape_features(box_resized)
            
            # Save result
            results.append({
                'annotation_id': annotation.id,
                'image_id': image.id,
                'species': species_name,
                'filename': image.filename,
                'color_histogram': color_histogram,
                'shape_features': shape_features
            })
            
            # Save extracted box for visual verification
            output_path = os.path.join(output_dir, f"{species_name}_{annotation.id}.jpg")
            cv2.imwrite(output_path, box)
            print(f"  Saved extracted box to {output_path}")
            
        except Exception as e:
            print(f"  Error processing annotation {annotation.id}: {str(e)}")
    
    print(f"\nAnalysis complete:")
    print(f"Processed {len(results)} annotations successfully")
    print(f"Failed to extract {extraction_failures} annotations")
    
    # Save features to numpy file for further analysis
    if results:
        species_counts = {}
        for result in results:
            species = result['species']
            species_counts[species] = species_counts.get(species, 0) + 1
        
        print("\nFeatures extracted by species:")
        for species, count in species_counts.items():
            print(f"  {species}: {count} samples")
        
        # Save features data
        np.save(os.path.join(output_dir, "features_data.npy"), results)
        print(f"Feature data saved to {os.path.join(output_dir, 'features_data.npy')}")

def analyze_feature_differences(data_file, output_dir="./similarity_analysis"):
    """Analyze differences in features between species."""
    try:
        # Load data
        results = np.load(data_file, allow_pickle=True)
        
        if len(results) == 0:
            print("Error: No data found in the feature file")
            return
        
        # Group by species
        species_data = {}
        for result in results:
            species = result['species']
            if species not in species_data:
                species_data[species] = {
                    'color_histograms': [],
                    'shape_features': []
                }
            
            species_data[species]['color_histograms'].append(result['color_histogram'])
            species_data[species]['shape_features'].append(result['shape_features'])
        
        # Calculate average feature vectors for each species
        species_averages = {}
        for species, data in species_data.items():
            if len(data['color_histograms']) > 0:
                color_histograms = np.array(data['color_histograms'])
                shape_features = np.array(data['shape_features'])
                
                species_averages[species] = {
                    'avg_color_histogram': np.mean(color_histograms, axis=0),
                    'avg_shape_features': np.mean(shape_features, axis=0),
                    'std_color_histogram': np.std(color_histograms, axis=0),
                    'std_shape_features': np.std(shape_features, axis=0)
                }
        
        # Compare feature differences
        print("\nFeature Differences:")
        species_list = list(species_averages.keys())
        for i in range(len(species_list)):
            for j in range(i+1, len(species_list)):
                species1 = species_list[i]
                species2 = species_list[j]
                
                print(f"\n{species1} vs {species2}:")
                
                # Color histogram difference
                color_hist_diff = np.abs(
                    species_averages[species1]['avg_color_histogram'] - 
                    species_averages[species2]['avg_color_histogram']
                ).mean()
                
                # Shape features difference
                shape_features_diff = np.abs(
                    species_averages[species1]['avg_shape_features'] - 
                    species_averages[species2]['avg_shape_features']
                ).mean()
                
                print(f"  Color histogram difference: {color_hist_diff:.4f}")
                print(f"  Shape features difference: {shape_features_diff:.4f}")
                
                # Calculate most distinguishing features
                color_diff_normalized = np.abs(
                    species_averages[species1]['avg_color_histogram'] - 
                    species_averages[species2]['avg_color_histogram']
                ) / (species_averages[species1]['std_color_histogram'] + 
                     species_averages[species2]['std_color_histogram'] + 0.001)
                
                shape_diff_normalized = np.abs(
                    species_averages[species1]['avg_shape_features'] - 
                    species_averages[species2]['avg_shape_features']
                ) / (species_averages[species1]['std_shape_features'] + 
                     species_averages[species2]['std_shape_features'] + 0.001)
                
                print("  Most distinguishing color features (hue bins):")
                top_color_indices = np.argsort(color_diff_normalized[:30])[::-1][:3]
                for idx in top_color_indices:
                    print(f"    Hue bin {idx}: {color_diff_normalized[idx]:.2f} times std dev")
                
                print("  Most distinguishing shape features:")
                shape_features = ["Area", "Perimeter", "Aspect ratio", "Extent", "HuMoment1", "HuMoment2"]
                top_shape_indices = np.argsort(shape_diff_normalized)[::-1][:3]
                for idx in top_shape_indices:
                    if idx < len(shape_features):
                        print(f"    {shape_features[idx]}: {shape_diff_normalized[idx]:.2f} times std dev")
        
    except Exception as e:
        print(f"Error analyzing feature differences: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Animal similarity analysis tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Extract features command
    extract_parser = subparsers.add_parser("extract", help="Extract features from species annotations")
    extract_parser.add_argument("--species", nargs="+", help="Species names to analyze (default: Wolf, Jackal)")
    extract_parser.add_argument("--output", default="./similarity_analysis", help="Output directory")
    
    # Analyze features command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze feature differences between species")
    analyze_parser.add_argument("--data", required=True, help="Path to features data file (.npy)")
    analyze_parser.add_argument("--output", default="./similarity_analysis", help="Output directory")
    
    args = parser.parse_args()
    
    app = create_app()
    with app.app_context():
        if args.command == "extract":
            analyze_species_similarities(args.species, args.output)
        elif args.command == "analyze":
            analyze_feature_differences(args.data, args.output)
        else:
            parser.print_help()

if __name__ == "__main__":
    main()