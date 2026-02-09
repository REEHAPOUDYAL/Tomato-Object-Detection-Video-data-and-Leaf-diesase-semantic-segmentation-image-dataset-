# Tomato Leaf Disease Segmentation (Semantic Segmentation)

## Project Overview
This research project focuses on **precision detection and semantic segmentation** of three tomato leaf diseases:

- **Early Blight**
- **Late Blight**
- **Leaf Miner**

Unlike basic classification approaches, this work implements **multiclass semantic segmentation** to provide **pixel-level localization** of diseased regions. This enables spatial understanding of infection spread, supporting **early-stage agricultural intervention and disease monitoring**.

### Key Achievements
- **Dataset**: Curated and managed **10,000 data** using **Roboflow**
- **Accuracy**: Achieved **97.03% Multiclass Pixel Accuracy** ** 0.81 IoU** 
- **Localization Performance**: Achieved **0.8107 mean IoU (Intersection over Union)**

---

## Research & Methodology
This project followed an **iterative research lifecycle**, evolving from baseline segmentation architectures into a customized high-performance pipeline.

---

###  Architectural Evolution

#### Baseline Models
Initially, the following architectures were tested:

- **U-Net + ResNet50 Encoder**
- **U-Net + ResNet34 Encoder**

However, these models showed limitations in effectively capturing **multi-scale features**, especially for **small and scattered disease spots**.

#### Optimal Architecture (Final Model)
**DeepLabV3+ with MobileNetV2 Encoder**

---

### Why MobileNetV2?
MobileNetV2 was selected because it provides:

- **Efficient inference**
- **Lower parameter count**
- **Reduced risk of overfitting**
- Strong feature extraction capability for specialized agricultural datasets

This made it an optimal encoder backbone for high-accuracy segmentation with lightweight performance.

---

## Handling Class Imbalance (Research Challenge)
A major challenge during training was the dominance of **background pixels**, which initially biased the model and reduced performance on disease regions.

### Custom Patching Strategy
To solve this issue, a custom preprocessing pipeline was developed:

- High-resolution tomato leaf images were divided into **3×3 patches**
- **Pure background patches were automatically removed**
- This significantly improved the foreground-background ratio and enhanced training stability

---

### Composite Loss Function
A weighted loss function was designed to ensure accurate segmentation boundaries:

- **Focal Loss**
  - Handles class imbalance by down-weighting easy background pixels  
- **Dice Loss**
  - Optimizes overlap between prediction and ground truth masks  
  - Improves boundary precision

Final Loss Function: **Weighted (Focal + Dice Loss)**

---

## Technical Implementation

###  Training Framework
- **PyTorch**


### Optimization Setup
- **Optimizer**: AdamW  
- **Scheduler**: ReduceLROnPlateau  

### Transfer Learning Strategy
- Used a pretrained **ImageNet MobileNetV2 encoder**
- Performed fine-tuning by unfreezing the **last 18 layers** of the encoder

### Evaluation Metrics
Model performance was evaluated using:

- **Multiclass IoU**
- **Dice Score**
- **F1 Score**
- **Multiclass Pixel Accuracy**

---

## Deployment & Reproducibility
To bridge research and real-world deployment, the project was engineered for portability and reproducibility.

### ONNX Conversion
The trained PyTorch model was exported to **ONNX format** for:

- Framework-agnostic deployment
- High-performance inference

### Dockerization
A full Docker environment was created to ensure consistent deployment across:

- Local machines
- Cloud instances
- Servers

### Streamlit User Interface
A **Streamlit-based dashboard** was developed to support:

- Real-time image uploads
- Automated inference
- Visualization of segmentation masks

---

## Performance Metrics
| Metric | Score |
|--------|-------|
| Multiclass Pixel Accuracy | **97.03%** |
| Mean IoU (IoU) | **0.8107** |
| Loss | **0.1975** |
| F1 score| **0.9688** |
| Dice score | **0.8724** |

---


