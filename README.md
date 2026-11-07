# Computer Vision & Deep Learning Framework for Fruit Quality Grading and Physical Trait Extraction

<p align="center">
  <img src="figures/01_contour_dimensioning_results.png" alt="Contour Dimensioning and Trait Extraction" width="850">
</p>

<p align="center">
  <a href="https://github.com/arafat678/4-th-sem-code/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/OpenCV-4.5%2B-green?logo=opencv" alt="OpenCV">
  <img src="https://img.shields.io/badge/scikit--learn-1.0%2B-F7931E?logo=scikit-learn" alt="scikit-learn">
  <img src="https://img.shields.io/badge/FastAI-2.5%2B-blueviolet" alt="FastAI">
  <img src="https://img.shields.io/badge/Thesis-M.Tech%204th%20Sem-success" alt="M.Tech 4th Sem">
</p>

---

## 📌 Abstract & Research Background

This repository contains the official codebase and experimental findings for the **M.Tech 4th Semester Thesis Research Project** on automated post-harvest fruit quality evaluation, physical morphological trait extraction, and multi-class defect grading using computer vision and deep learning.

Manual agricultural grading in commercial sorting sheds is labor-intensive, subjective, and prone to high error margins. This research implements an end-to-end cyber-physical inspection pipeline that integrates **classical morphological computer vision** (calibrated contour tracking, convex hull computation, bounding-box aspect ratio extraction, HSV color thresholding) with **modern deep convolutional neural networks** (custom 3-stage ConvNet and transfer-learned ResNet-50 / ResNet-34 architectures). The pipeline achieves **98.7% classification accuracy** across multi-tier commercial quality grades (Premium, Grade A, Grade B, Reject) while maintaining real-time processing throughput exceeding **70 FPS**.

---

## 🔬 Key Research Findings & Visual Gallery

### 1. Physical Trait & Morphology Extraction
The system isolates individual fruit contours, fits approximating convex hulls, and computes equatorial width, polar height, bounding box dimensions, and surface solidity metrics in real time.

| Calibrated Measurement Overlay | Grayscale Luminance Map | Binary Inverted Mask |
|:---:|:---:|:---:|
| <img src="figures/output_image-1.jpg" width="270" alt="Annotated Measurement"> | <img src="figures/output_image-2.jpg" width="270" alt="Grayscale Transform"> | <img src="figures/output_image-3.jpg" width="270" alt="Binary Mask"> |
| *Figure 1(a): Width/Height bounding box and convex hull overlay* | *Figure 1(b): Median-filtered grayscale transform* | *Figure 1(c): Inverse threshold ($T=210$) segmentation mask* |

---

### 2. Surface Defect Localization & Classification Performance
High-resolution defect contour tracking maps surface blemishes, scabs, and bruising, feeding into the deep learning classifier.

| Defect Contour Segmentation | Multi-Class Confusion Matrix | Training & Validation Curves |
|:---:|:---:|:---:|
| <img src="figures/02_defect_contour_detection.png" width="270" alt="Defect Contours"> | <img src="figures/03_confusion_matrix.png" width="270" alt="Confusion Matrix"> | <img src="figures/04_training_accuracy_loss.png" width="270" alt="Training Curves"> |
| *Figure 2: Segmented defect regions & damage contours* | *Figure 3: Confusion matrix across 4 grading tiers* | *Figure 4: Convergence history across 30 training epochs* |

---

### 3. Varietal Prediction & Grading Output

| Varietal Prediction Confidence | Web Evaluation Dashboard | Complete Pipeline Output |
|:---:|:---:|:---:|
| <img src="figures/05_multi_variety_prediction.png" width="270" alt="Variety Confidence"> | <img src="figures/06_web_application_dashboard.png" width="270" alt="Web Dashboard"> | <img src="figures/07_quality_grading_output.png" width="270" alt="Pipeline Output"> |
| *Figure 5: Softmax probability across cultivars* | *Figure 6: Interactive sorting dashboard* | *Figure 7: Unified quality tier classification* |

---

## 📊 Empirical Performance Benchmark

Comprehensive empirical comparison between five machine learning and deep learning architectures evaluated on the same benchmark test set:

| Architecture | Model Family | Parameters | Accuracy | Precision | Recall | F1-Score | Latency (ms) | Throughput (FPS) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ResNet-50 (Fine-Tuned)** | Transfer Learning | ~25.6M | **98.7%** | **98.6%** | **98.7%** | **0.986** | 14.2 ms | 70.4 FPS |
| **ResNet-34 (Fine-Tuned)** | Transfer Learning | ~21.8M | 97.4% | 97.2% | 97.4% | 0.973 | 9.8 ms | 102.0 FPS |
| **Custom ConvNet (3-Stage)** | Convolutional Net | ~0.92M | 94.2% | 93.8% | 94.2% | 0.940 | **4.1 ms** | **243.9 FPS** |
| **Linear SVM Classifier** | Support Vector Machine | N/A | 89.5% | 89.1% | 89.5% | 0.893 | 2.3 ms | 434.7 FPS |
| **k-NN Classifier ($k=3$)** | Instance-Based | N/A | 84.8% | 84.2% | 84.8% | 0.845 | 5.7 ms | 175.4 FPS |

---

## 🧠 System Architecture & Methodology Flow

```mermaid
flowchart TD
    A["Raw Input RGB Image / Live Video Feed"] --> B["Image Preprocessing & Median Blur (k=5)"]
    B --> C["Binary Inverse Thresholding (T=210)"]
    C --> D["Topological Border Following (Suzuki Contour Algorithm)"]
    
    subgraph Trait_Extraction["Physical Trait Extraction"]
        D --> E["Bounding Box Dimensioning: Width (W), Height (H)"]
        D --> F["Aspect Ratio Calculation (AR = W / H)"]
        D --> G["Polygon Approximation (Douglas-Peucker) & Convex Hull"]
        D --> H["Solidity Computation: Area(C) / Area(Hull)"]
        D --> I["Defect Area Ratio (DAR) Extraction"]
    end
    
    subgraph Color_Analysis["Color Space Analysis"]
        B --> J["RGB to HSV Conversion"]
        J --> K["Multi-Color Range Masking (Red, Yellow, Green)"]
        J --> L["Interactive Trackbar Calibration"]
    end
    
    subgraph Classification_Engine["Multi-Class Grading & Classification"]
        A --> M["Spatial Normalization (150x150 / 224x224)"]
        M --> N1["Custom 3-Stage ConvNet (Batch Normalization)"]
        M --> N2["ResNet-50 / ResNet-34 (FastAI Backbone)"]
        M --> N3["SVM with Standard Scaling"]
    end
    
    Trait_Extraction --> O["Unified Decision Matrix & Grade Assignment"]
    Color_Analysis --> O
    Classification_Engine --> O
    
    O --> P1["Premium (Grade 1)"]
    O --> P2["Grade A (Grade 2)"]
    O --> P3["Grade B (Grade 3)"]
    O --> P4["Reject / Industrial Defective"]
```

---

## 📐 Mathematical Formulation

### 1. External Contour and Convex Hull Approximation
Given an extracted discrete contour boundary $C = \{p_1, p_2, \dots, p_N\}$, polygon approximation is evaluated with epsilon threshold $\epsilon$:
$$\epsilon = 0.0001 \times \text{arcLength}(C, \text{closed}=\text{True})$$
$$\text{approx} = \text{approxPolyDP}(C, \epsilon, \text{closed}=\text{True})$$

The Convex Hull $\text{Conv}(C)$ defines the minimal convex envelope:
$$\text{Conv}(C) = \left\{ \sum_{i=1}^N \lambda_i p_i \;\middle|\; \lambda_i \ge 0, \sum_{i=1}^N \lambda_i = 1 \right\}$$

### 2. Physical Solidity & Aspect Ratio
$$\text{Solidity} = \frac{\text{Area}(C)}{\text{Area}(\text{Conv}(C))}, \qquad \text{Aspect Ratio} = \frac{W}{H}$$
where $W$ and $H$ represent the orthogonal bounds of the minimum bounding rectangle.

### 3. Objective Loss Function (Multi-Class Cross-Entropy)
$$\mathcal{L}_{\text{CE}}(\Theta) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log \left( \frac{\exp(z_{i,c})}{\sum_{k=1}^C \exp(z_{i,k})} \right) + \frac{\lambda}{2} \|\Theta\|_2^2$$

---

## 📂 Repository File Structure

```text
4-th-sem-code/
├── figures/                              # High-resolution research result figures & charts
│   ├── 01_contour_dimensioning_results.png
│   ├── 02_defect_contour_detection.png
│   ├── 03_confusion_matrix.png
│   ├── 04_training_accuracy_loss.png
│   ├── 05_multi_variety_prediction.png
│   ├── 06_web_application_dashboard.png
│   ├── 07_quality_grading_output.png
│   ├── output_image-1.jpg                # Bounding box & hull dimension overlay
│   ├── output_image-2.jpg                # Grayscale transform
│   └── output_image-3.jpg                # Inverted binary mask
├── src/                                  # Modular source package
│   ├── trait_extraction/
│   │   ├── __init__.py
│   │   └── physical_metrics.py           # Dimensions, convex hull, and contour metrics
│   ├── classification/
│   │   ├── __init__.py
│   │   ├── cnn_model.py                  # PyTorch ConvNet architecture
│   │   ├── svm_classifier.py             # Linear/RBF SVM classifier
│   │   └── knn_classifier.py             # k-NN baseline classifier
│   └── color_analysis/
│       ├── __init__.py
│       ├── color_tuner.py                # Interactive HSV trackbar tuner
│       └── multi_color_mask.py           # Multi-color range segmentation
├── docs/                                 # Detailed scientific documentation
│   ├── METHODOLOGY.md                    # In-depth mathematical and computer vision modeling
│   └── BENCHMARK_ANALYSIS.md             # Class-wise ablation studies & latency benchmarks
├── main.py                               # Unified command-line interface & pipeline runner
├── requirements.txt                      # Python dependencies
├── CITATION.cff                          # Formal academic citation metadata
├── LICENSE                               # MIT License
└── README.md                             # Repository documentation
```

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/arafat678/4-th-sem-code.git
cd 4-th-sem-code
pip install -r requirements.txt
```

### 2. Extract Physical Traits from an Image
```bash
python main.py --task extract --image figures/output_image-1.jpg --output_dir figures
```

### 3. Display Multi-Model Benchmark Comparison
```bash
python main.py --task benchmark
```

### 4. Launch Interactive HSV Color Space Tuner
```bash
python main.py --task color_tune --image figures/output_image-1.jpg
```

---

## 📖 Citation

If this research, pipeline, or dataset analysis assists your work, please cite it as:

```bibtex
@misc{mondal2024fruitqualitygrading,
  author       = {Arafat Ali Mondal},
  title        = {Computer Vision and Deep Learning Framework for Fruit Quality Grading and Physical Trait Extraction},
  year         = {2024},
  publisher    = {GitHub},
  journal      = {GitHub Repository},
  howpublished = {\url{https://github.com/arafat678/4-th-sem-code}},
  note         = {M.Tech 4th Semester Thesis Research Project, Maulana Abul Kalam Azad University of Technology (MAKAUT)}
}
```

---

## 📜 License
This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
