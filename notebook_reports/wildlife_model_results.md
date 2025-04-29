# Wildlife Detection Model Results

## Training Summary
- **Model**: YOLOv8n (nano size) - optimized for small datasets
- **Dataset**: 358 training images, 90 validation images
- **Classes**: 28 wildlife species including deer, foxes, rabbits and various other animals
- **Training epochs**: 100
- **Image size**: 640x640
- **Hardware**: NVIDIA GeForce RTX 4050 Laptop GPU with CUDA 12.4

## Performance Metrics
- **Best mAP50-95**: 0.225
- **Best mAP50**: 0.391
- **Precision**: 0.245
- **Recall**: 0.560

## Class-specific Performance
| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| Male Roe Deer | 0.245 | 0.857 | 0.718 | 0.409 |
| Female Roe Deer | 0.040 | 0.500 | 0.047 | 0.027 |
| Fox | 0.128 | 0.333 | 0.230 | 0.109 |
| Rabbit | 0.187 | 0.667 | 0.689 | 0.470 |
| Blackbird | 0.000 | 0.000 | 0.000 | 0.000 |
| Human | 0.562 | 1.000 | 0.663 | 0.335 |

## Observations
- The model performs best on Male Roe Deer, Rabbits, and Humans
- The model struggles with rare species (Blackbird) and species with limited training examples
- The model requires very low confidence thresholds (0.01-0.05) to detect wildlife
- False positives frequently appear in areas with leaf patterns and shadows
- The low mAP scores (0.225) indicate the model is not production-ready and needs:
  - More training data, especially for underrepresented species
  - Potential class balancing techniques
  - Possibly a larger model architecture

## Next Steps
1. Collect more annotated wildlife images
2. Address class imbalance with techniques like class weights, oversampling, or data augmentation
3. Experiment with model architecture (YOLOv8s or YOLOv8m)
4. Implement test-time augmentation for improved detection
5. Explore domain-specific pre-training on similar wildlife datasets

## Training Progress
The model improved steadily until around epoch 80-90, with final validation metrics of mAP50 = 0.391, which is still quite low for reliable wildlife detection. This suggests the model needs substantially more training data or a different approach to handle the complexity of wildlife detection in natural habitats.