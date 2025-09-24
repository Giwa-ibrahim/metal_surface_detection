# Metal Surface Defect Detection using CNNs Optimized with Gazelle Optimization Algorithms

A deep learning project for metal surface defect classification using Convolutional Neural Networks (CNNs) optimized with Gazelle Optimization Algorithms (GOA) and their advanced variants. It features extensive image preprocessing, dataset transformation, and a comparative analysis of multiple optimization methods for improved defect detection in industrial quality control.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Methodology](#methodology)
  - [About the Dataset](#about-the-dataset)
  - [Data Preprocessing](#data-preprocessing)
  - [Dataset Transformation](#dataset-transformation)
  - [Model Architecture & Training](#model-architecture--training)
- [Optimization Algorithms Implemented](#optimization-algorithms-implemented)
- [Model Evaluation & Results](#model-evaluation--results)
- [Comparison Table](#comparison-table)
- [How to Use](#how-to-use)
- [Applications](#applications)
- [Appendix: Abbreviations](#appendix-abbreviations)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

This project implements a robust pipeline for detecting and classifying metal surface defects, leveraging advanced CNN architectures and a suite of metaheuristic optimization algorithms. The goal is to automate and improve the reliability of industrial quality control processes.

---

## Methodology

### About the Dataset

- **Source:** Kaggle
- **Samples:** 1,800 grayscale images, categorized into six common defects: rolled-in scale (RS), patches (Pa), crazing (Cr), pitted surface (PS), inclusion (In), and scratches (Sc).
- **Structure:** Three folders—`train`, `test`, and `valid`—each with six subfolders (one per defect class). The `train` folder contains 276 images per class, while `test` and `valid` folders have 12 images per class.

### Data Preprocessing

- **Adaptive Image Sharpening:** Enhanced edge features using a Gaussian filter.
- **CLAHE:** Improved image contrast.
- *Other techniques (such as bilateral filtering, edge detection, and contour extraction) were tested but not adopted due to negligible improvement.*

### Dataset Transformation

- Images converted to 3-channel grayscale (for model compatibility)
- Resized to 256x256 pixels
- Normalized to [-1, 1]
- Prepared as PyTorch tensors and loaded via DataLoader for training, validation, and testing splits

### Model Architecture & Training

- **Model:** TinyVGG-2 (lightweight VGG-inspired CNN)
- **Core Layers:** Two convolutional blocks, each with Conv2D, ReLU, and MaxPool2D, followed by a fully connected classifier.
- **Parameters:** ~375k trainable
- **Input:** 3x256x256 images

The model was trained with various optimizers (detailed below), with hyperparameters tuned via randomized search.

---

## Optimization Algorithms Implemented

- **Standard GOA**
- **Levy Flight GOA**
- **Roulette Wheel GOA**
- **Random Walk GOA**
- **Adaptive GOA**
- **Particle Swarm Optimization (PSO)**
- **Genetic Algorithm (GA)**
- **Predator Aware Momentum GOA (PAM-GOA) [Proposed]**

Each optimizer features unique strategies for hyperparameter search, exploration, and exploitation, and was evaluated using the same CNN backbone for fair comparison.

---

## Model Evaluation & Results

**Metrics used:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix.

Each optimizer's run includes full classification reports and confusion matrices (see notebooks for details).

---

## Comparison Table

| Model                    | Training Acc. | Validation Acc. | Test Acc. | Test Macro F1 |
|--------------------------|---------------|-----------------|-----------|--------------|
| CNN with Standard GOA    | 97.48%        | 98.61%          | 87.50%    | 0.877        |
| CNN with Levy Flight GOA | 98.92%        | 95.83%          | 90.28%    | 0.902        |
| CNN with Roulette GOA    | 99.22%        | 93.06%          | 88.89%    | 0.888        |
| CNN with Random Walk GOA | 98.56%        | 95.83%          | 88.89%    | 0.890        |
| CNN with Adaptive GOA    | 98.56%        | 91.67%          | 84.72%    | 0.848        |
| CNN with PSO             | 97.78%        | 91.67%          | 80.56%    | 0.799        |
| CNN with GA              | 99.70%        | 94.44%          | 87.50%    | 0.869        |
| **CNN with PAM-GOA**     | **99.34%**    | **98.61%**      | **91.67%**| **0.890**    |

---

## How to Use

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Giwa-ibrahim/metal_surface_detection.git
   ```
2. **Install Dependencies:**  
   Ensure Python and Jupyter are installed, then install dependencies as specified in the notebooks (e.g., PyTorch, NumPy, OpenCV).
3. **Run the Notebooks:**  
   Open the Jupyter notebooks and execute sequentially to reproduce preprocessing, model training, and result analysis.

---

## Applications

- **Industrial Quality Control:** Automated, objective surface defect detection.
- **Benchmarking:** Comparative study of metaheuristic optimizers in computer vision.
- **Research:** Extendable for testing new optimization strategies and architectures.

---

## Appendix: Abbreviations

- **GOA:** Gazelle Optimization Algorithm
- **PSO:** Particle Swarm Optimization
- **GA:** Genetic Algorithm
- **PAM-GOA:** Predator Aware Momentum GOA (proposed)
- See full list of optimizer-specific parameters at the end of the main notebook.

---

## Acknowledgements

- Kaggle for dataset provision.
- Open-source tools (PyTorch, NumPy, OpenCV).
- Research community for algorithm inspirations and baseline implementations.

---

For questions or contributions, please open an issue or submit a pull request.
