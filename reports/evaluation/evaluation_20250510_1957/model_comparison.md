# Wildlife Detection Model Comparison

Date: 2025-05-10 20:12:26

## Performance Comparison

### Standard vs. Hierarchical Model

![Model Comparison](model_comparison.png)

### Detailed Metrics

#### Confidence Threshold = 0.25

| Metric | Standard | Hierarchical | Improvement |
|--------|----------|--------------|-------------|
| precision | 0.4810 | 0.8969 | 86.4% |
| recall | 0.5488 | 0.7860 | 43.2% |
| mAP50 | 0.5187 | 0.8911 | 71.8% |
| mAP50-95 | 0.3303 | 0.5853 | 77.2% |

#### Confidence Threshold = 0.5

| Metric | Standard | Hierarchical | Improvement |
|--------|----------|--------------|-------------|
| precision | 0.5462 | 0.8885 | 62.7% |
| recall | 0.3725 | 0.7941 | 113.2% |
| mAP50 | 0.4649 | 0.8697 | 87.1% |
| mAP50-95 | 0.3060 | 0.5832 | 90.6% |

## Key Findings

The **hierarchical model outperforms the standard model** in the majority of metrics. The most significant improvement is in **recall** at threshold 0.5, with an increase of **113.2%**.

This supports the hypothesis that grouping species into taxonomic categories improves detection performance, especially for wildlife with limited training examples.

## Recommendations

1. **Adopt the hierarchical approach** for wildlife detection as the primary method.
2. Consider a two-stage detection pipeline, where the hierarchical model identifies the taxonomic group, followed by a specialized model to identify the specific species within that group.
3. Continue collecting additional training data, particularly for rare species with few examples.
4. Explore model ensembling techniques to combine the strengths of both approaches.
