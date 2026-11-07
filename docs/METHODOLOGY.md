# Technical Methodology: Fruit Quality Grading & Trait Extraction

## 1. Overview
This document outlines the computer vision, mathematical morphology, and deep learning methodologies developed for the M.Tech 4th semester thesis research project on automated fruit quality grading and physical trait extraction.

---

## 2. Image Preprocessing Pipeline

### 2.1 Grayscale Conversion & Noise Filtering
Given an RGB fruit image $I(x, y) \in \mathbb{R}^{H \times W \times 3}$, conversion to single-channel intensity space $I_{\text{gray}}(x, y)$ is performed via standard luminance weighting:
$$I_{\text{gray}}(x, y) = 0.299 \cdot R(x, y) + 0.587 \cdot G(x, y) + 0.114 \cdot B(x, y)$$

To attenuate high-frequency acquisition noise (e.g., sensor salt-and-pepper artifacts) while preserving sharp fruit boundary gradients, a non-linear median filter of kernel size $k \times k$ ($k = 5$) is applied:
$$I_{\text{blur}}(x, y) = \text{median} \{ I_{\text{gray}}(x + i, y + j) \mid (i, j) \in [-2, 2]^2 \}$$

### 2.2 Adaptive Binary Inverse Segmentation
Segmentation between fruit foreground and illuminated background is achieved through inverted binary thresholding:
$$B(x, y) = \begin{cases} 255, & \text{if } I_{\text{blur}}(x, y) < T_{\text{cutoff}} \\ 0, & \text{otherwise} \end{cases}$$
where optimal empirical boundary separation is achieved at $T_{\text{cutoff}} = 210$.

---

## 3. Physical Trait & Geometric Modeling

### 3.1 Bounding Box & Aspect Ratio (AR)
External contours $C = \{p_1, p_2, \dots, p_N\}$ are extracted using Suzuki's topological border following algorithm (`cv2.findContours` with `RETR_EXTERNAL` and `CHAIN_APPROX_SIMPLE`).

From the bounding box with coordinates $(x_{\min}, y_{\min})$ and dimensions $(W, H)$:
$$W = x_{\max} - x_{\min}, \quad H = y_{\max} - y_{\min}$$
$$AR = \frac{W}{H}$$
Fruit sphericity is benchmarked against optimal roundness ($AR \approx 1.00 \pm 0.08$).

### 3.2 Convex Hull & Solidity
The convex hull $\text{Conv}(C)$ represents the smallest convex set containing contour $C$:
$$\text{Conv}(C) = \left\{ \sum_{i=1}^N \lambda_i p_i \;\middle|\; \lambda_i \ge 0, \sum_{i=1}^N \lambda_i = 1 \right\}$$

The physical solidity $S$ measures boundary indentation (identifying stem cavities and mechanical indentation defects):
$$S = \frac{\text{Area}(C)}{\text{Area}(\text{Conv}(C))}$$

### 3.3 Defect Area Ratio (DAR)
Defect detection isolates darker bruised or decayed patches $D(x, y)$ on the fruit surface:
$$DAR = \frac{\sum_{x, y} D(x, y)}{\sum_{x, y} B(x, y)} \times 100\%$$

---

## 4. Multi-Class Deep Learning Architecture

### 4.1 Custom ConvNet Pipeline
A 3-stage convolutional neural network processes normalized input tensors $\mathbf{X} \in \mathbb{R}^{3 \times 150 \times 150}$:
- **Block 1:** $\text{Conv2D}(3 \to 12, 3 \times 3) \to \text{BatchNorm} \to \text{ReLU} \to \text{MaxPool}(2 \times 2) \implies \mathbb{R}^{12 \times 75 \times 75}$
- **Block 2:** $\text{Conv2D}(12 \to 20, 3 \times 3) \to \text{ReLU} \implies \mathbb{R}^{20 \times 75 \times 75}$
- **Block 3:** $\text{Conv2D}(20 \to 32, 3 \times 3) \to \text{BatchNorm} \to \text{ReLU} \implies \mathbb{R}^{32 \times 75 \times 75}$
- **Classifier:** $\text{Flatten} \to \text{Dense}(180,000 \to 5) \to \text{Softmax}$

### 4.2 Loss Function & Optimization
Training minimizes the Multi-Class Cross-Entropy loss:
$$\mathcal{L}_{\text{CE}} = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C y_{i, c} \log(\hat{y}_{i, c})$$
Optimized using the Adam algorithm ($\eta = 10^{-3}, \beta_1 = 0.9, \beta_2 = 0.999$, weight decay $\lambda = 10^{-4}$).

### 4.3 Transfer Learning with ResNet-50
Pretrained ImageNet weights are adapted via progressive fine-tuning (discriminative layer learning rates):
$$\eta_{\text{base}} = 10^{-4}, \quad \eta_{\text{head}} = 10^{-3}$$
yielding superior feature extraction across varied apple cultivars (Fuji, McIntosh, Scarlet, Tydemans Early).
