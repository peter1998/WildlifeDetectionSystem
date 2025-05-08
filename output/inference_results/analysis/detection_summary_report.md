# Wildlife Detection Summary Report

**Generated on:** 2025-05-05 18:42:44

## Model Information
- **Model Path:** /home/peter/Desktop/TU PHD/WildlifeDetectionSystem/models/trained/wildlife_detector_20250503_1345/weights/best.pt
- **Model Type:** best
- **Confidence Threshold:** 0.1
- **IoU Threshold:** 0.7
- **Device:** cpu

## Overall Statistics
- **Total Images:** 20
- **Images with Detections:** 7 (35.0%)
- **Total Detections:** 9
- **Total Processing Time:** 0.77 seconds
- **Average Processing Time per Image:** 0.0387 seconds

## Species Distribution
| Species | Count | Percentage |
|---------|-------|------------|
| Human | 3 | 33.3% |
| Female Roe Deer | 3 | 33.3% |
| Male Roe Deer | 2 | 22.2% |
| Rabbit | 1 | 11.1% |

## Taxonomic Group Distribution
| Group | Count | Percentage |
|-------|-------|------------|
| Deer | 5 | 55.6% |
| Other | 3 | 33.3% |
| Small_Mammals | 1 | 11.1% |

## Light Condition Analysis
| Condition | Images | Percentage |
|-----------|--------|------------|
| daylight | 5 | 25.0% |
| twilight | 15 | 75.0% |

## Confidence Analysis
- **Average Confidence:** 0.597
- **Median Confidence:** 0.677
- **Min Confidence:** 0.119
- **Max Confidence:** 0.865

## Bounding Box Size Analysis
- **Average Area:** 1285207.3 pixels²
- **Average Width:** 909.4 pixels
- **Average Height:** 1176.7 pixels
- **Average Aspect Ratio (w/h):** 0.77

## Top 5 Largest Detections
| Image | Species | Confidence | Area (pixels²) |
|-------|---------|------------|----------------|
| 0878_x_IMAG0001.JPG | Human | 0.119 | 5165280 |
| 1252_15_03_24_Моллова_курия_IMAG0246.JPG | Female Roe Deer | 0.828 | 1741244 |
| 0040_16_06_IMAG0007.JPG | Human | 0.757 | 947646 |
| 0011_28_05_IMAG0148.JPG | Male Roe Deer | 0.387 | 758778 |
| 0011_28_05_IMAG0148.JPG | Female Roe Deer | 0.677 | 758664 |

## Top 5 Highest Confidence Detections
| Image | Species | Confidence | Area (pixels²) |
|-------|---------|------------|----------------|
| 1291_15_03_24_Моллова_курия_IMAG0385.JPG | Rabbit | 0.865 | 449457 |
| 1252_15_03_24_Моллова_курия_IMAG0246.JPG | Female Roe Deer | 0.828 | 1741244 |
| 0706_30_4_24_100BMCIM_IMAG0009.JPG | Human | 0.801 | 656586 |
| 0040_16_06_IMAG0007.JPG | Human | 0.757 | 947646 |
| 0011_28_05_IMAG0148.JPG | Female Roe Deer | 0.677 | 758664 |

## Detection Rates by Light Condition and Taxonomic Group
| Taxonomic Group | daylight | twilight |
|----------------|--- | --- |
| Deer | 100.0% | 0.0% |
| Other | 33.3% | 66.7% |
| Small_Mammals | 0.0% | 100.0% |

## Recommendations


## Analysis Images

The following analysis images were generated and saved in the output directory:

1. `species_distribution.png` - Distribution of detections by species
2. `taxonomic_distribution.png` - Distribution of detections by taxonomic group
3. `species_by_light_condition.png` - Species detection by light condition
4. `confidence_distribution.png` - Distribution of detection confidence scores
5. `bbox_size_distribution.png` - Distribution of bounding box sizes
6. `species_confidence.png` - Confidence distribution by species
7. `taxonomic_vs_light.png` - Heatmap of taxonomic groups by light condition
