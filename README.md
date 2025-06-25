# Multimodal Contrastive Learning for Complex System Anomaly Detection (MCLD)

## Overview

This repository implements a novel multimodal contrastive learning approach for complex system anomaly detection and fault identification. Our method leverages both temporal and frequency domain representations of time series data to achieve superior anomaly detection performance in industrial systems.

## Key Innovation

Our approach transforms traditional univariate anomaly detection into a multimodal learning problem by:

1. **Multi-Modal Representation**: Converting 1D time series data into 2D time-frequency representations using wavelet transforms, creating two complementary modalities
2. **Contrastive Learning Framework**: Employing contrastive learning to capture relationships between temporal and spectral features
3. **Joint Optimization**: Combining classification and contrastive losses for robust feature learning

## Methodology

### Architecture Overview

```
Time Series Data → Wavelet Transform → Time-Frequency Image
       ↓                                      ↓
   CNN Encoder 1D                         CNN Encoder 2D
       ↓                                      ↓
   Temporal Features                   Time-Frequency Features
       ↓                                      ↓
       └──────── Feature Fusion ──────────────┘
                        ↓
              Classification Head + Contrastive Loss
```

### Core Components

1. **Multimodal Data Generation**
   - Apply wavelet transform to time series data to generate time-frequency representations
   - Original 1D time series serves as temporal modality
   - Generated 2D time-frequency images serve as spectral modality

2. **Dual CNN Encoders**
   - **Encoder 1**: Extracts temporal features from 1D time series data
   - **Encoder 2**: Extracts spectral features from 2D time-frequency images
   - Both encoders output flattened feature vectors

3. **Joint Loss Function**
   - **Classification Loss**: Features are concatenated (torch.cat, dim=1) and passed through a classification head for cross-entropy loss
   - **Contrastive Loss**: Computed between temporal and spectral feature representations
   - **Total Loss**: L_total = L_classification + L_contrastive

## Performance

- **Dataset**: CWRU Bearing Dataset
- **Accuracy**: 99%
- **Framework**: PyTorch

## Dataset Information

This project is validated on the [CWRU Bearing Dataset](https://www.kaggle.com/datasets/brjapon/cwru-bearing-datasets), which contains:

### Experimental Setup
- **Motor**: 2 HP power with torque transducer and dynamometer
- **Test Conditions**: 1 HP load, 1772 rpm shaft speed
- **Sampling**: 48 kHz accelerometer sampling frequency
- **Sensors**: 3 accelerometers positioned at Drive End (DE), Fan End (FE), and Base (BA)

### Fault Types
- **Defect Locations**: Ball, inner race, outer race
- **Defect Sizes**: 0.007" (0.178mm), 0.014" (0.356mm), 0.021" (0.533mm)
- **Defect Introduction**: Single-point defects created by EDM machining

### Data Characteristics
- **Segment Length**: 2048 points (0.04 seconds at 48kHz)
- **Features**: 9 statistical features including maximum, minimum, mean, standard deviation, RMS, skewness, kurtosis, crest factor, and form factor

## Applications

This methodology is particularly suitable for:
- **Industrial Equipment Monitoring**: Early fault detection in rotating machinery
- **Predictive Maintenance**: Identifying potential failures before they occur
- **Quality Control**: Monitoring manufacturing processes for anomalies
- **Infrastructure Health**: Monitoring bridges, buildings, and other structures

## Key Advantages

- **Multimodal Learning**: Leverages both temporal and frequency domain information
- **High Accuracy**: Achieves 99% accuracy on benchmark dataset
- **Robust Feature Learning**: Contrastive learning improves feature discriminability
- **End-to-End Training**: Joint optimization of classification and contrastive objectives
- **Industrial Applicability**: Validated on real-world bearing fault data

## Technical Requirements

- **Framework**: PyTorch
- **Key Dependencies**: 
  - PyTorch
  - NumPy
  - SciPy (for wavelet transforms)
  - Matplotlib (for visualization)
  - Scikit-learn (for metrics)

## Citation


## License

[Your chosen license - MIT, Apache 2.0, etc.]

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.
