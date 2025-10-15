# 🕳️ Pothole Detection using CNN & Transfer Learning

This project focuses on building an automated deep learning system to detect **potholes** from road surface images.  
We compare multiple models — a **Baseline CNN** and several **Transfer Learning architectures** — and visualize model interpretability using **Grad-CAM**.

---

## 📑 Table of Contents
1. [Imports & Setup](#step-1-imports--setup)  
2. [Dataset Paths](#step-2-dataset-paths)  
3. [Count Images & Visualization](#step-3-count-images--visualization)  
4. [Data Generators](#step-4-data-generators)  
5. [Baseline CNN](#step-5-baseline-cnn)  
6. [Transfer Learning](#step-6-transfer-learning)  
7. [Train Transfer Models](#step-7-train-transfer-models)  
8. [Plot Training Results](#step-8-plot-training-results)  
9. [Summary of All Models](#step-9-summary-of-all-models)  
10. [Plot Summary Bars for Accuracy & Loss](#step-10-plot-summary-bars-for-accuracy--loss)  
11. [Compare All Models with Full Metrics](#step-11-compare-all-models-with-full-metrics)  
12. [Grad-CAM Visualization](#step-12-grad-cam-visualization)

---

## 🧠 Step 1: Imports & Setup
All essential libraries such as TensorFlow, Keras, OpenCV, NumPy, Matplotlib, and Scikit-learn are imported.  
GPU is enabled for faster training on large models.

---

## 📂 Step 2: Dataset Paths
Dataset used:  
`/kaggle/input/pothole-detection-dataset`

Dataset contains images categorized into:
- **Pothole**
- **No Pothole**

Each image is resized and augmented for better generalization.

---

## 🔢 Step 3: Count Images & Visualization
Counts the number of images in each class and visualizes sample images using Matplotlib.

---

## ⚙️ Step 4: Data Generators
Image data generators are used to:
- Normalize pixel values
- Apply augmentations (rotation, shift, brightness, shear, etc.)
- Split dataset into **training**, **validation**, and **test** sets

---

## 🧱 Step 5: Baseline CNN
A custom **Convolutional Neural Network** is built and trained from scratch.  
This model serves as a baseline for comparison with transfer learning models.

---

## 🔁 Step 6: Transfer Learning
Pre-trained models used:
- **VGG16**
- **MobileNetV2**
- **EfficientNetB0**

All models are fine-tuned on the pothole dataset with custom top layers.

---

## 🚀 Step 7: Train Transfer Models
Each model is trained using the same data pipeline and hyperparameters for a fair comparison.

---

## 📊 Step 8: Plot Training Results
Training and validation **accuracy** and **loss curves** are plotted for all models.

---

## 📜 Step 9: Summary of All Models
Summarizes all trained models with their **accuracy**, **precision**, **recall**, and **F1-score** metrics.

---

## 📈 Step 10: Plot Summary Bars for Accuracy & Loss
A bar plot comparison of all models’ performance metrics is shown for a quick overview.

---

## 🧩 Step 11: Compare All Models with Full Metrics
All models are compared in a single DataFrame with metrics including:
- Accuracy
- Precision
- Recall
- F1-score

---

## 🔥 Step 12: Grad-CAM Visualization
Grad-CAM heatmaps are generated for:
- Baseline CNN  
- VGG16  
- MobileNetV2  
- EfficientNetB0  
- Ensemble Model  

These visualizations highlight which image regions contributed most to the model’s decision.

---

## 🏆 Results Summary
| Model | Accuracy | Precision | Recall | F1-Score |
|--------|-----------|------------|---------|-----------|
| Custom CNN | ~96.99% | - | - | - |
| VGG16 | ~97.38% | - | - | - |
| MobileNetV2 | ~97.62% | - | - | - |
| EfficientNetB0 | ~98.93% | - | - | - |
| **Ensemble Model** | **98.93%** | - | - | - |

---

## 🧩 Technologies Used
- TensorFlow / Keras  
- OpenCV  
- NumPy & Pandas  
- Matplotlib / Seaborn  
- Scikit-learn  

---

## 📁 Dataset
The dataset can be found on Kaggle:  
🔗 [Pothole Detection Dataset](https://www.kaggle.com/datasets)

---

## 🧠 Author
**Abdullah Al Mahmud Joy**  
Kaggle Notebook: [Potholes Detection with Baseline CNN & TL](https://www.kaggle.com/code/abdullahalmahmudjoy/potholes-detections-with-baseline-cnn-tl)

---

## 📌 License
This project is released under the **MIT License** — free to use and modify with attribution.

---

## 💡 Future Work
- Real-time pothole detection using YOLOv8 or SSD  
- Integration with GPS tagging for road repair systems  
- Deployment as a web or mobile app
