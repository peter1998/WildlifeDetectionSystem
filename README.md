# Wildlife Detection System

An AI-powered system for detecting, annotating, and analyzing wildlife in camera trap images. The project combines user-friendly annotation interfaces with a robust backend API and database system to facilitate creation of training datasets for machine learning models.

## Project Overview

This system allows researchers to:
- Manage large collections of camera trap images
- Annotate wildlife with bounding boxes and species labels
- Export annotations in standard formats (YOLO, COCO) for ML training
- Track annotation progress across multiple datasets
- Analyze wildlife behavior patterns and habitat usage

## Features

- **Image Management**: Index, upload, and organize camera trap images
- **Species Cataloging**: Maintain database of wildlife species with metadata
- **Annotation Interfaces**: Multiple UI options for bounding box creation
- **Export Functionality**: Generate datasets in YOLO and COCO formats
- **Microhabitat Analysis**: Analyze vegetation and environmental conditions
- **Behavioral Tracking**: Monitor wildlife activity patterns

## System Architecture

```
WildlifeDetectionSystem/
├── api/                  # Flask backend
│   ├── app/              # Application source
│   │   ├── models/       # Database models
│   │   ├── routes/       # API endpoints
│   │   ├── services/     # Business logic
│   │   ├── static/       # Static files (annotation UIs)
│   │   ├── templates/    # HTML templates
│   │   └── utils/        # Helper utilities
│   ├── config.py         # Configuration
│   ├── debug/            # Debugging tools
│   ├── instance/         # Instance data (contains SQLite DB)
│   └── run.py            # Application entry point
├── data/                 # Data storage
│   ├── raw_images/       # Original camera trap images
│   ├── processed_images/ # Standardized images
│   ├── annotations/      # Annotation data directory
│   └── export/           # Export directory for ML data
├── docs/                 # Documentation
├── scripts/              # Utility scripts
└── requirements.txt      # Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.8+
- pip
- SQLite

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/WildlifeDetectionSystem.git
   cd WildlifeDetectionSystem
   ```

2. Create a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Set up the database
   ```bash
   cd api
   export FLASK_APP=run.py
   flask shell
   >>> from app.models.models import db
   >>> db.create_all()
   >>> exit()
   ```

5. Run the server
   ```bash
   flask run
   ```

6. Access the annotation interface
   Open a web browser and navigate to: http://127.0.0.1:5000/advanced-annotator

## Usage

### Annotating Images

1. Select a folder from the dropdown
2. Draw a box around wildlife by clicking and dragging
3. Select the species from the right panel
4. Adjust the box if needed
5. Navigate between images with Previous/Next buttons
6. Mark "No Animals Present" for empty images
7. Submit annotations with the "Submit & Next" button

### Keyboard Shortcuts

- B - Box tool
- S - Select tool
- Delete - Delete annotation
- F - Fit to screen
- +/- - Zoom in/out
- 1-9 - Select species
- Left/Right arrows - Navigate images

### Exporting Data

Use the API endpoints:
- COCO format: GET /api/annotations/export
- YOLO format: GET /api/annotations/export/yolo?output_dir=./data/export

## Development Roadmap

### Phase 1: Basic Recognition
- Identify complete animals in good conditions
- Integrate with existing labeled image database
- Basic differentiation of target species

### Phase 2: Advanced Recognition
- Recognize partial silhouettes and characteristic features
- Adapt to various imaging conditions
- Integrate environmental factors

### Phase 3: Complex Analysis
- Diurnal and seasonal activity
- Species interaction tracking
- Behavioral pattern analysis

## Documentation

Detailed documentation can be found in the [docs](./docs) directory.

## Contributing

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Project developed in collaboration with Prof. Peeva
- Inspired by needs in wildlife conservation research
- Based on camera trap technology for non-invasive wildlife monitoring
