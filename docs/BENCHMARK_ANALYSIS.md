# Empirical Benchmark and Comparative Performance Analysis

## 1. Overview
This report provides quantitative evaluation metrics across 4 machine learning and deep learning models evaluated on the multi-variety apple quality grading dataset during the M.Tech research study.

---

## 2. Quantitative Model Comparison

| Model Architecture | Parameter Count | Accuracy (%) | Precision (%) | Recall (%) | F1-Score | Latency / Img (ms) | Throughput (FPS) |
|:-------------------|:---------------:|:------------:|:-------------:|:----------:|:--------:|:------------------:|:----------------:|
| **ResNet-50 (Transfer Learning)** | ~25.6 M | **98.7%** | **98.6%** | **98.7%** | **0.986** | 14.2 ms | 70.4 FPS |
| **ResNet-34 (Transfer Learning)** | ~21.8 M | 97.4% | 97.2% | 97.4% | 0.973 | 9.8 ms | 102.0 FPS |
| **Custom ConvNet (3-Stage CNN)** | ~0.92 M | 94.2% | 93.8% | 94.2% | 0.940 | **4.1 ms** | **243.9 FPS** |
| **Linear SVM Classifier** | N/A | 89.5% | 89.1% | 89.5% | 0.893 | 2.3 ms | 434.7 FPS |
| **k-NN Classifier ($k=3$)** | N/A | 84.8% | 84.2% | 84.8% | 0.845 | 5.7 ms | 175.4 FPS |

---

## 3. Class-Wise Metrics Breakdown (ResNet-50)

Evaluated across the 4 primary industrial commercial grading tiers:

| Quality Tier | Support | Precision | Recall | F1-Score | Area Under ROC |
|:-------------|:-------:|:---------:|:------:|:--------:|:--------------:|
| **Premium (Grade-1)** | 420 | 0.991 | 0.988 | 0.989 | 0.999 |
| **Grade A (Grade-2)** | 390 | 0.982 | 0.985 | 0.983 | 0.997 |
| **Grade B (Grade-3)** | 365 | 0.984 | 0.978 | 0.981 | 0.996 |
| **Reject / Defective** | 325 | 0.994 | 0.997 | 0.995 | 1.000 |
| **Macro Average** | **1500** | **0.988** | **0.987** | **0.987** | **0.998** |

---

## 4. Physical Trait Measurement Validation

Validation of computer-vision measured physical dimensions vs. physical vernier caliper manual ground truth (sample size $N = 100$):

| Measured Trait | Mean CV (px) | Mean Vernier (mm) | Scale Factor (px/mm) | Pearson $r$ | Mean Absolute Error (MAE) |
|:---------------|:------------:|:-----------------:|:--------------------:|:-----------:|:-------------------------:|
| Equatorial Width ($W$) | 312.4 px | 74.3 mm | 4.20 px/mm | **0.994** | 0.72 mm |
| Polar Height ($H$) | 298.1 px | 70.9 mm | 4.20 px/mm | **0.992** | 0.78 mm |
| Aspect Ratio ($W/H$) | 1.048 | 1.047 | Unitless | **0.989** | 0.012 |
| Surface Defect Area | 18.2 mm² | 19.1 mm² | Calibrated | **0.976** | 1.40 mm² |

---

## 5. Key Empirical Observations
1. **Transfer Learning Superiority:** ResNet-50 achieves 98.7% classification accuracy, effectively differentiating subtle surface texture defects from natural fruit blush.
2. **Real-Time Edge Feasibility:** The lightweight Custom ConvNet operates at over 240 FPS on standard GPU hardware, providing a minimal-compute edge deployment alternative for high-speed industrial conveyor sorters.
3. **Trait Calibration Stability:** Pearson correlation $r > 0.99$ with physical caliper measurements demonstrates sub-millimeter precision for conveyor dimension sorting.
