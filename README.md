# **🚗 Real-Time Object Detection for Autonomous Vehicles**

### **Project 4 \- AI & Data Science Track**

## **📝 Project Overview**

This repository contains the end-to-end source code and documentation for building a **Real-Time Object Detection System** designed for autonomous driving. The project was executed in five milestones, covering data acquisition, model training, optimization, and MLOps deployment.

The system uses a **YOLOv8** model fine-tuned on the **BDD100K dataset** to detect dynamic obstacles (vehicles, pedestrians, riders) while filtering out static environmental noise.

## **📂 Repository Structure**

├── 01\_Data\_Prep/  
│   ├── 1\_Exploratory\_Data\_Analysis.ipynb  \# M1: Dataset analysis & visualization  
│   └── 2\_Preprocessing\_Augmentation.ipynb \# M1: Resizing & normalization scripts  
│  
├── 02\_Training/  
│   ├── 3\_YOLOv8\_Training.ipynb            \# M2: Transfer learning workflow  
│   └── runs/                              \# Training logs and confusion matrices  
│  
├── 03\_Deployment/  
│   ├── export\_to\_onnx.py                  \# M3: Conversion script (PT \-\> ONNX)  
│   ├── test\_and\_record.py                 \# M3: Local inference script (with HUD)  
│   └── best.onnx                          \# Final optimized model  
│  
├── 04\_Monitoring\_Dashboard/               \# M4: Streamlit App  
│   ├── app.py                             \# Main Dashboard code  
│   ├── requirements.txt                   \# App dependencies  
│   └── packages.txt                       \# System dependencies  
│  
├── 05\_Reports/  
│   ├── Milestone1\_Data\_Report.pdf  
│   ├── Milestone2\_Evaluation\_Report.pdf  
│   ├── Milestone3\_Testing\_Report.pdf  
│   └── Milestone4\_MLOps\_Report.pdf  
│  
└── README.md

## **📅 Milestones & Deliverables**

### **✅ Milestone 1: Data Collection & Preprocessing**

**Objective:** Prepare a diverse autonomous driving dataset.

* **Dataset:** [BDD100K (Berkeley DeepDrive)](https://www.google.com/search?q=https://bdd-data.berkeley.edu/)  
* **Activities:** \* Exploratory Data Analysis (EDA) on class distribution.  
  * Image resizing to $640 \\times 640$.  
  * Data augmentation (brightness, horizontal flip) to simulate weather conditions.  

### **✅ Milestone 2: Model Development**

**Objective:** Train and evaluate multiple efficient object detection models to select the best performer.

Comparative Analysis:  
We evaluated three state-of-the-art YOLO architectures to determine the optimal balance between accuracy and speed.

| Model | Environment | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Notes |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **YOLOv12** | Google Colab (Tesla T4) | 0.4336 | 0.2417 | 0.6120 | 0.3850 | Attention-centric; 13 FPS inference. |
| **YOLOv9-M** | Kaggle (Tesla T4) | 0.5492 | 0.3158 | 0.7322 | 0.4947 | Robust detection capabilities. |
| **YOLOv11-M** | Kaggle (Tesla T4) | **0.5519** | **0.3186** | **0.7390** | 0.4946 | **Best Performer** (High Precision). |

* **Selected Model:** YOLOv11-M (referred to as YOLOv8/11 pipeline in deployment) due to superior mAP and precision scores.  
* **Training Specs:** Transfer Learning from COCO, 50 Epochs, SGD Optimizer.  

### **✅ Milestone 3: Deployment & Testing**

**Objective:** Optimize model for real-time edge inference.

* **Optimization:** Exported to **ONNX** format for CPU acceleration.  
* **Inference Speed:** optimized to **\~14 FPS** on standard CPU (Ryzen 7\) using $320 \\times 320$ input resizing.  
* **Logic:** Implemented a "Ban List" to filter non-obstacle classes (lanes) for cleaner output.  

### **✅ Milestone 4: MLOps & Monitoring**

**Objective:** Create a dashboard to monitor system health in production.

* **Tool:** Streamlit \+ WebRTC.  
* **Features:**  
  * Real-time Camera Streaming (Mobile/Desktop).  
  * Live Telemetry (FPS, Latency, Object Count).  
  * Drift Detection Alerts (if Confidence \< 40%).  
* https://autonomous-vehicle-demogit.streamlit.app/

### **✅ Milestone 5: Final Documentation**

* **Deliverable:** Final Project Report summarizing the entire lifecycle.  
* **Presentation:** Slide deck demonstrating the live dashboard.


## **📊 Key Results**

| Metric | Value | Notes |  
| Model | YOLOv12 | Fine-tuned on BDD100K |  
| Inference Format | ONNX | OpsSet 12 |  
| Avg Latency | 76ms | Tested on CPU |  
| Detection Speed | \~13.6 FPS | Real-time capable |

