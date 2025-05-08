# Wildlife Detection Model Evaluation Report

## Model Information
- **Model**: YOLOv8 Wildlife Detector
- **Evaluation Date**: 2025-05-05 18:42:22
- **Model Path**: /home/peter/Desktop/TU PHD/WildlifeDetectionSystem/models/trained/wildlife_detector_20250505_1800/weights/best.pt
- **Dataset Path**: /home/peter/Desktop/TU PHD/WildlifeDetectionSystem/data/export/yolo_default_20250429_085945/data.yaml

## Performance Metrics (conf=0.25)
- **mAP50-95**: 0.2058
- **mAP50**: 0.3556
- **Precision**: 0.3287
- **Recall**: 0.4143

## Performance by Confidence Threshold
| Threshold | mAP50 | Precision | Recall |
|-----------|-------|-----------|--------|
| 0.50 | 0.3538 | 0.3366 | 0.4132 |
| 0.25 | 0.3556 | 0.3287 | 0.4143 |
| 0.10 | 0.3567 | 0.3287 | 0.4143 |
| 0.05 | 0.3575 | 0.4537 | 0.4143 |

## Taxonomic Group Performance
| Group | Precision | Recall | F1 Score | Samples |
|-------|-----------|--------|----------|--------|
| Deer | 0.9216 | 0.9216 | 0.9216 | 113 |
| Small_Mammals | 1.0000 | 0.8276 | 0.9057 | 60 |
| Other | 0.7500 | 1.0000 | 0.8571 | 21 |
| Carnivores | 0.3846 | 0.7143 | 0.5000 | 19 |
| Birds | 0.0000 | 0.0000 | 0.0000 | 2 |

## Observations and Recommendations
- The model shows best balance between precision and recall at threshold 0.05
- For wildlife detection applications, consider using a threshold of 0.05-0.1 for higher recall
- For scientific analysis requiring high confidence, use thresholds of 0.3-0.5
- The model performs best on the Deer taxonomic group
- The Birds group shows lower performance and may benefit from additional training data
- Implement a hierarchical detection approach using taxonomic groups for first-stage detection
- Consider ensemble methods for improved accuracy on challenging species
