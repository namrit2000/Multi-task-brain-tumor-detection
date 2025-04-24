# 🧠 Multi-Task Brain Tumor Detection Using Deep Learning

This project presents a unified deep learning architecture for the classification and segmentation of brain tumors using contrast-enhanced T1-weighted MRI scans. The study investigates multi-task learning (MTL) with a ResNet50 backbone and compares it against a segmentation-only baseline using EfficientNetB0.

---

## 🎯 Project Objective

To develop a scalable deep learning model capable of:
- Classifying tumor types (Meningioma, Glioma, Pituitary)
- Segmenting tumor regions in MRI scans  
...using a shared encoder in a **multi-task learning framework**.

---

## 🧬 Dataset

- **Source**: [Figshare Brain Tumor Dataset](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)
- **Total Samples**: 3,064 grayscale MRI slices
- **Annotations**:  
  - Class labels: `meningioma`, `glioma`, `pituitary`  
  - Pixel-wise binary tumor masks

---

## 🧪 Models Implemented

### 1. EfficientNetB0 U-Net (Segmentation Only)
- Encoder: EfficientNetB0
- Decoder: Custom transpose conv stack
- Loss: Dice + Binary Cross Entropy  
- Outcome: High Mean IoU (1.00) but lacked classification; showed instability

### 2. ResNet50 Multi-Task Model (Classification + Segmentation)
- Shared Encoder: Pretrained ResNet50
- Dual Heads:
  - **Classification**: Fully connected + Softmax
  - **Segmentation**: Decoder + Sigmoid
- Losses:
  - Categorical Crossentropy (classification)
  - Binary Crossentropy (segmentation)
- Optimizer: Adam
- Regularization: Early stopping, class weights for imbalance

---

## ⚙️ Preprocessing Pipeline

- Converted `.mat` files to `.png`
- Resized to 224×224, normalized to [0,1]
- Replicated grayscale channels to RGB for pretrained ResNet
- One-hot encoded class labels
- Stratified 80/20 train-validation split

---

## 📈 Performance Summary

| Model | Classification Accuracy | Segmentation Accuracy | Notes |
|-------|--------------------------|------------------------|-------|
| **ResNet50 Multi-Task** | 75.69% | 98.31% | Balanced performance, interpretable |
| **EfficientNetB0 U-Net** | N/A | 98.31% (IoU=1.0) | Metric instability, lacked classification |

### Confusion Matrix (ResNet50):
- Meningioma: Precision 0.60, Recall 0.78  
- Glioma: Precision 0.80, Recall 0.72  
- Pituitary: Precision 0.90, Recall 0.79

---

## 📊 Visualizations

- ROC curves for all classes  
- Segmentation masks (predicted vs. ground truth)  
- Confusion matrices  
- Loss and accuracy training curves

---

## 🔍 Key Takeaways

- Multi-task learning enhances efficiency and interpretability for medical diagnosis.
- ResNet50 shared encoder generalizes well across dual tasks with limited data.
- EfficientNetB0 underperformed without domain-specific tuning despite its speed.
