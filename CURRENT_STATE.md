# Wildlife Detection System - Current State Documentation
*Generated on April 1, 2025*

## System Overview

The Wildlife Detection System is an AI-powered platform for detecting, annotating, and analyzing wildlife in camera trap images. The system combines user-friendly annotation interfaces with a robust backend API for creating training datasets for machine learning models.

## Current Status

- **Database Status**:
  - Images indexed: 1,882
  - Species defined: 28 (including "Background" species)
  - Annotations created: 54 total
    - 9 actual wildlife annotations
    - 45 "Background" annotations (for images with no animals)

- **Interfaces**:
  - Advanced Annotator: Main working interface for creating bounding box annotations
  - Simple Annotator: Simplified alternative interface
  - Ground Truth Annotator: Alternative interface with different layout
  - **New** Dashboard: Central hub for accessing all system functions
  - **New** Environmental Editor: Interface for adding environmental metadata

- **Key Functionality**:
  - Image indexing and browsing
  - Bounding box annotation with species labeling
  - "No Animals Present" marking
  - Export to COCO and YOLO formats
  - Environmental data tracking (light conditions, habitat type)

## System Architecture

```
WildlifeDetectionSystem/
├── api/                  # Flask backend
│   ├── app/              # Application source
│   │   ├── models/       # Database models
│   │   ├── routes/       # API endpoints
│   │   ├── services/     # Business logic
│   │   ├── static/       # Static files
│   │   ├── templates/    # HTML templates (Dashboard, Environmental Editor)
│   │   └── utils/        # Helper utilities
│   ├── config.py         # Configuration
│   ├── debug/            # Debugging tools
│   ├── instance/         # Instance data (SQLite DB)
│   ├── scripts/          # Utility scripts
│   └── run.py            # Application entry point
├── data/                 # Data storage
│   ├── raw_images/       # Original camera trap images
│   │   └── test_01/      # First dataset (~1882 images)
│   ├── processed_images/ # Standardized images
│   ├── annotations/      # Annotation data directory
│   └── export/           # Export directory for ML data
```

## Database Schema

### Key Tables
- **Image**: Stores metadata about camera trap images
- **Species**: Contains wildlife species definitions
- **Annotation**: Stores bounding box annotations linking images to species
- **EnvironmentalData**: Stores environmental factors for each image

## API Endpoints

### Core Endpoints
- `GET /api/images/`: List all images
- `GET /api/species/`: List all species
- `GET /api/annotations/image/<id>`: Get annotations for an image
- `POST /api/annotations/`: Create new annotation
- `POST /api/annotations/batch`: Save multiple annotations for an image
- `POST /api/images/<id>/no-animals`: Mark image as having no animals
- `GET /api/annotations/export`: Export annotations in COCO format
- `GET /api/annotations/export/yolo`: Export annotations in YOLO format

### Environmental Data Endpoints
- `GET /api/environmental/image/<id>`: Get environmental data for an image
- `POST /api/environmental/image/<id>`: Create/update environmental data

## Web Interfaces

### Main Interfaces
- **Dashboard** (`/`): Central hub for accessing all functions
- **Advanced Annotator** (`/advanced-annotator`): Main annotation interface
- **Environmental Editor** (`/environmental-editor`): Environmental data editor

## Annotation Workflow

1. Access the Advanced Annotator at http://127.0.0.1:5000/advanced-annotator
2. Select a folder from the dropdown
3. Browse images and either:
   - Draw bounding boxes around wildlife and select species
   - Click "No Animals Present" for empty images
4. Click "Submit & Next" to save and proceed to the next image

## Environmental Data Workflow

1. Access the Environmental Editor at http://127.0.0.1:5000/environmental-editor
2. Select a folder and image
3. Add environmental metadata:
   - Light condition (Full darkness, Early twilight, Late twilight, Daylight)
   - Temperature
   - Snow cover
   - Habitat type
   - Vegetation type
4. Click "Save Data" to store the information

## Data Export

Export options are available from the Dashboard:
- COCO format: Exports annotations in COCO JSON format
- YOLO format: Exports annotations in YOLO format with corresponding files

## Current Issues and Limitations

1. Limited number of real annotations (only 9 wildlife annotations so far)
2. Environmental data needs to be populated for existing images
3. No authentication system (single-user environment)

## Next Steps (Based on Prof. Peeva's Requirements)

1. **Data Collection**:
   - Prof. Peeva will provide additional data after April 13, 2025
   - Continue annotating test_01 dataset (1,873 images remaining)

2. **Environmental Data**:
   - Add environmental metadata for all annotated images
   - Focus on light conditions (darkness, twilight, daylight)
   - Track snow cover and habitat types

3. **Analysis Development**:
   - Implement seasonal/annual activity analysis (not monthly)
   - Develop tools for analyzing species in blurry/partial images
   - Work on differentiating similar species (wolf/jackal)

## Running the System

```bash
cd /home/peter/Desktop/TU\ PHD/WildlifeDetectionSystem/api
source venv/bin/activate
export FLASK_APP=run.py
export FLASK_DEBUG=1
flask run
```

Access the system at http://127.0.0.1:5000/