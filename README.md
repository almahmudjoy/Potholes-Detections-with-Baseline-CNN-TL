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
<img width="457" height="398" alt="image" src="https://github.com/user-attachments/assets/90dbe875-7ff4-4c7d-bbc2-6fea3b2f63ac" />
<img width="469" height="368" alt="image" src="https://github.com/user-attachments/assets/c00abecc-e705-4009-9110-985f9a7b525f" />

---

## ⚙️ Step 4: Data Generators
Image data generators are used to:
- Normalize pixel values
- Apply augmentations (rotation, shift, brightness, shear, etc.)
- Split dataset into **training**, **validation**, and **test** sets

<img width="458" height="78" alt="image" src="https://github.com/user-attachments/assets/ac0a2a50-c2e4-4d34-a5c0-d1c98611d829" />

---

## 🧱 Step 5: Baseline CNN
A custom **Convolutional Neural Network** is built and trained from scratch.  
This model serves as a baseline for comparison with transfer learning models.

---

## 🔁 Step 6: Transfer Learning
Fine-tuning of pre-trained models:
- **VGG16**
- **MobileNetV2**
- **EfficientNetB0**
- **ResNet50V2**

Custom dense layers are added for classification.

---

## 🚀 Step 7: Train Transfer Models
Each model is trained with identical hyperparameters and data pipelines for fair comparison.

---

## 📊 Step 8: Plot Training Results
Training and validation **accuracy** and **loss curves** are plotted for all models.
<img width="696" height="318" alt="image" src="https://github.com/user-attachments/assets/506d153d-6cb4-42f2-8c07-68de2a06a0e3" />
<img width="704" height="311" alt="image" src="https://github.com/user-attachments/assets/890208f0-027d-4eb9-8e7b-790a7c3e8f0f" />
<img width="677" height="336" alt="image" src="https://github.com/user-attachments/assets/2b6e4c98-efda-45cb-a678-4bdc0e9efcfa" />
<img width="684" height="314" alt="image" src="https://github.com/user-attachments/assets/a9f9a7e1-5b9d-4ff4-8797-7855dc918dcd" />
<img width="699" height="320" alt="image" src="https://github.com/user-attachments/assets/bbbc0d7e-62e5-44d8-bb34-601bc79760e3" />

---

## 📜 Step 9: Summary of All Models
All models are evaluated on the test set using accuracy and loss.
| Model | Loss | Accuracy |
|:------------------|:----------:|:----------:|
| **Baseline CNN** | 1.188794 | 0.518519 |
| **VGG16 (Fine-Tuned)** | 0.064942 | **0.985185** |
| **MobileNetV2 (Fine-Tuned)** | 0.078864 | 0.962963 |
| **EfficientNetB0 (Fine-Tuned)** | 0.691336 | 0.511111 |
| **ResNet50V2 (Fine-Tuned)** | **0.049485** | **0.985185** |

✅ **Top Performers:**  
- **VGG16 FT** and **ResNet50V2 FT** both achieved **98.5% accuracy**  
- **MobileNetV2 FT** achieved **96.2% accuracy**

---

## 📈 Step 10: Plot Summary Bars for Accuracy & Loss
A bar plot comparison of all models’ performance metrics is shown for a quick overview.

**Baseline CNN - Confusion Matrix & ROC Curve**

<img width="695" height="420" alt="image" src="https://github.com/user-attachments/assets/797603d7-86ed-4580-9e21-1aae8bb73d5e" />
<img width="681" height="415" alt="image" src="https://github.com/user-attachments/assets/b4602448-3f3c-4c2b-bf5c-de5096d20565" />

**VGG16 - Confusion Matrix & ROC Curve**

<img width="429" height="392" alt="image" src="https://github.com/user-attachments/assets/da3dd700-1682-434c-9b9d-56016b05c583" />
<img width="488" height="389" alt="image" src="https://github.com/user-attachments/assets/d85fb92e-1293-4791-a700-b42d81cd8a88" />

**MobileNetV2 - Confusion Matrix & ROC Curve**

<img width="449" height="387" alt="image" src="https://github.com/user-attachments/assets/43019106-d0bf-48b1-b163-2021dca7efa7" />
<img width="490" height="387" alt="image" src="https://github.com/user-attachments/assets/ff629cee-401a-4ef6-86b7-966b4b61bbac" />

**EfficientNetB0 - Confusion Matrix & ROC Curve**

<img width="469" height="394" alt="image" src="https://github.com/user-attachments/assets/e85bad9c-51db-4d7e-a988-f3d7a6d1793c" />
<img width="477" height="379" alt="image" src="https://github.com/user-attachments/assets/e4d74ce2-8b71-4748-a8a0-b690484e5e47" />

**ResNet50V2 - Confusion Matrix & ROC Curve**

<img width="423" height="391" alt="image" src="https://github.com/user-attachments/assets/f0bfcf8a-d401-4aec-8509-b78320d16cc6" />
<img width="492" height="391" alt="image" src="https://github.com/user-attachments/assets/e3b2d250-2730-4f39-8647-aaf8e798de68" />

---

## 🧩 Step 11: Compare All Models with Full Metrics
All models are compared in a single DataFrame with metrics :
<img width="461" height="378" alt="image" src="https://github.com/user-attachments/assets/c961882c-b325-4ffd-89bc-b5b4f2f32a4c" />
<img width="444" height="374" alt="image" src="https://github.com/user-attachments/assets/4d96d1db-024a-4d95-b6ac-65d087acd23f" />


---

## 🔥 Step 12: Grad-CAM Visualization
Grad-CAM heatmaps show the most influential regions for model predictions on road images.  
This helps interpret model decisions and confirm that pothole regions are correctly attended.

<img width="625" height="314" alt="image" src="https://github.com/user-attachments/assets/e9495a9c-d96c-4b49-ad5e-0eaa9adc0e29" />
<img width="626" height="299" alt="image" src="https://github.com/user-attachments/assets/3c0e0ab5-aeb0-4a08-a6b9-338b632f48b7" />
<img width="624" height="295" alt="image" src="https://github.com/user-attachments/assets/6a61589a-4184-4c75-9c93-eee3e6516bdd" />
<img width="643" height="311" alt="image" src="https://github.com/user-attachments/assets/b8d2985d-a3d2-47cf-b4e5-431b085d746f" />
<img width="618" height="292" alt="image" src="https://github.com/user-attachments/assets/01011c1b-f58f-4115-bcd7-70057164a14c" />

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
🔗 [Pothole Detection Dataset](https://www.kaggle.com/datasets/atulyakumar98/pothole-detection-dataset)

---

## 🧠 Author
**Abdullah Al Mahmud Joy**  
- [Kaggle Notebook](https://www.kaggle.com/code/abdullahalmahmudjoy/potholes-detections-with-baseline-cnn-tl)

---

💡 *Created by [Abdullah Al Mahmud Joy](https://www.kaggle.com/abdullahalmahmudjoy)*  
B.Sc. in CSE | Teaching Assistant at BUBT  | AI & Deep Learning Enthusiast  
