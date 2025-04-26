# UNITE_FAU_Thesis: Solder Inspection for Pseudo Error Reduction in PCB Production Using Unsupervised Learning

This repository contains the code, experiments, and documentation for the master's thesis project: **Solder Inspection for Pseudo Error Reduction in Printed Circuit Board (PCB) Production using Unsupervised Learning**. The project was conducted at **Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)** in cooperation with **Siemens AG**, Berlin, as part of the requirements for the M.Sc. in Data Science.

The thesis investigates the use of unsupervised anomaly detection techniques — particularly leveraging the **Anomalib** library — to automate the identification of faulty solder joints on PCB assemblies, aiming to reduce dependency on manual labeling and improve scalability in industrial quality control.

---

## Table of Contents
- [Project Motivation](#project-motivation)
- [Dataset](#dataset)
- [Methods and Models](#methods-and-models)
- [Experiments](#experiments)
- [Results](#results)
- [How to Use This Repository](#how-to-use-this-repository)
- [Key Dependencies](#key-dependencies)
- [Acknowledgements](#acknowledgements)
- [License & Confidentiality](#license--confidentiality)
- [Contact](#contact)

---

## Project Motivation

Automated Optical Inspection (AOI) machines are widely used in PCB manufacturing for defect detection. However, these systems often generate a high number of false calls, requiring costly manual intervention and expert labeling for supervised machine learning models. As PCB complexity grows, the need for scalable, label-free inspection methods becomes critical.

This project explores unsupervised anomaly detection approaches to:
- Eliminate the need for large, manually labeled datasets.
- Reduce false calls and manual intervention.
- Provide scalable, adaptable solutions for industrial PCB inspection.

---

## Dataset

- **Source**: Provided by Siemens AG, Berlin.
- **Content**: 6,014 images of solder joints on PCBs (512x512 pixels), split into:
  - **FC** (No Defects)
  - **NG** (Defective)
- **Splits**:
  - **Training**: 2,971 FC / 1,712 NG
  - **Testing**: 743 FC / 428 NG
  - **Custom Test**: 80 FC / 80 NG
- **Image Types**: Both colored and heatmap images.
- **Preprocessing**: Images resized to 256x256 for modeling.

---

## Methods and Models

### Baseline (Supervised)
- **YOLOv8**: State-of-the-art object detection model, trained with manually labeled data.

### Unsupervised Anomaly Detection (via Anomalib)
- **PatchCore**: Uses locally-aware patch features and memory banks for anomaly detection.
- **Deep Feature Modeling (DFM)**: Models normal data distribution using deep feature extraction and density modeling.
- **Deep Feature Kernel Density Estimation (DFKDE)**: Combines deep feature extraction with kernel density estimation.
- **FastFlow**: Employs 2D normalizing flows for efficient anomaly detection and localization.
- **EfficientAD**: Lightweight student-teacher approach for fast, resource-efficient anomaly detection.

### Feature Extractors
- **ResNet18, ResNet50, WideResNet50-2**
- **DenseNet121/169/201**
- **Vision Transformers (DeiT, CaiT)**

---

## Experiments

- **Data Loading**: Custom Anomalib datamodule for efficient preprocessing and splitting.
- **Model Training/Inference**: Each model is trained (or features extracted) on normal images only, with hyperparameter tuning for backbone, layers, and thresholds.
- **Evaluation Metrics**: AUROC, Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
- **Model Export**: Trained models exported in OpenVINO format for deployment.

---

## Results

| Model          | Accuracy | AUROC | Precision | Recall | F1-Score |
|----------------|----------|-------|-----------|--------|----------|
| YOLOv8-L       | 95%      | -     | 1.00      | 0.925  | 0.96     |
| YOLOv8-M       | 93.75%   | -     | 0.99      | 0.925  | 0.95     |
| PatchCore      | 91.25%   | 0.93  | 0.87      | 0.98   | 0.92     |
| DFM (best)     | 72.86%   | 0.72  | 0.65      | 0.97   | 0.78     |
| DFKDE (best)   | 70.71%   | 0.77  | 0.64      | 0.97   | 0.77     |
| FastFlow (ViT) | 66.87%   | 0.88  | 0.60      | 1.00   | 0.75     |
| EfficientAD    | 60%      | 0.67  | 0.56      | 1.00   | 0.71     |

**Key Finding**: PatchCore achieved the best unsupervised performance, with only a 2.5% gap to the supervised YOLOv8-M baseline.

**Challenges**: Some unsupervised models suffered from high false positive rates and longer training times, limiting real-time deployment.

**Future Work**: Further exploration of Vision Transformers and optimization for faster inference are recommended.

---

## How to Use This Repository

### Prerequisites
- Python 3.8+
- Anomalib
- PyTorch, OpenVINO, and other dependencies (see `requirements.txt`)

### Setup
```bash
# Clone the repository
git clone https://github.com/rohitpotdukhe01/UNITE_FAU_Thesis.git
cd UNITE_FAU_Thesis

# Install dependencies
pip install -r requirements.txt
```

---

## Key Dependencies

- [Anomalib](https://github.com/openvinotoolkit/anomalib)
- PyTorch
- OpenVINO
- scikit-learn
- numpy, pandas, matplotlib

---

## Acknowledgements

First and foremost, I would like to thank my supervisor, Dr. Majid Mortazavi, for his invaluable guidance, support, and encouragement throughout the course of my research. His insights and expertise were crucial to the successful completion of this work. 

I am also deeply grateful to my advisor, Prof. Dr. Enrique Zuazua, for his continuous support, constructive feedback, and for always being available to discuss ideas and provide guidance. 

I would like to extend my gratitude to Siemens AG, Berlin, Germany, for providing me with the opportunity to conduct my research in collaboration with their team and for providing the required resources. 

Finally, my heartfelt thanks go to my family and friends for their unwavering support, patience, and understanding throughout this journey. Their encouragement has been a source of strength for me.

---

## License & Confidentiality

**Note:** This repository contains or references confidential information and is subject to a non-disclosure agreement with Siemens AG. Duplication or publication of the thesis or any proprietary data/code is **not permitted** without prior written approval from Siemens AG.

---

## Contact

For questions regarding this project or collaboration, please contact:

- **Rohit Potdukhe (Author)**: [your.email@domain.com]
- **Prof. Dr. Enrique Zuazua (Advisor)**: [university contact]
- **Dr. Majid Mortazavi (Supervisor)**: [Siemens contact]

> *This README summarizes the core aspects of the thesis and provides practical guidance for reproducing or extending the experiments. For full technical details, please refer to the thesis document and code comments.*