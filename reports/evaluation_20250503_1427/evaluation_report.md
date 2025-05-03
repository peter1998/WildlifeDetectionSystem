# Wildlife Detection Model Evaluation Report

## Model Information
- **Model**: YOLOv8 Wildlife Detector
- **Evaluation Date**: 2025-05-03 14:27:40
- **Model Path**: /home/peter/Desktop/TU PHD/WildlifeDetectionSystem/models/trained/wildlife_detector_20250503_1345/weights/best.pt
- **Dataset Path**: /home/peter/Desktop/TU PHD/WildlifeDetectionSystem/data/export/yolo_default_20250429_085945/data.yaml

## Performance Metrics (conf=0.25)
- **mAP50-95**: 0.3686
- **mAP50**: 0.5336
- **Precision**: 0.5362
- **Recall**: 0.5032

## Performance by Confidence Threshold
| Threshold | mAP50 | Precision | Recall |
|-----------|-------|-----------|--------|
| 0.50 | 0.5057 | 0.5316 | 0.4405 |
| 0.25 | 0.5336 | 0.5362 | 0.5032 |
| 0.10 | 0.5453 | 0.7877 | 0.5138 |
| 0.05 | 0.6129 | 0.7877 | 0.5138 |

## Taxonomic Group Performance
| Group | Precision | Recall | F1 Score | Samples |
|-------|-----------|--------|----------|--------|
| Deer | 1.0000 | 0.9091 | 0.9524 | 51 |
| Other | 0.9000 | 1.0000 | 0.9474 | 11 |
| Small_Mammals | 0.8462 | 0.9167 | 0.8800 | 28 |
| Carnivores | 0.7895 | 0.7895 | 0.7895 | 13 |

## Observations and Recommendations
- The model shows best balance between precision and recall at threshold 0.10
- For wildlife detection applications, consider using a threshold of 0.05-0.1 for higher recall
- For scientific analysis requiring high confidence, use thresholds of 0.3-0.5
- The model performs best on the Deer taxonomic group
- The Carnivores group shows lower performance and may benefit from additional training data
- Implement a hierarchical detection approach using taxonomic groups for first-stage detection
- Consider ensemble methods for improved accuracy on challenging species
