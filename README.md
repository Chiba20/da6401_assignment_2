# DA6401 Assignment 2 – Building a Complete Visual Perception Pipeline

* **Name:** Nas-bah Khamis Issa
* **Course:** DA6401 – Introduction to Deep Learning

---
## Weights & Biases Report

Public W&B Report Link:  
https://api.wandb.ai/links/ge26z812-iitm-india/ftgsmccf
All experiments, visualizations, and analysis required for the assignment are in this report.

## Repository Description
Github repository link :
https://github.com/Chiba20/da6401_assignment_2.git
This repository contains a complete implementation of a **multi-stage visual perception pipeline** built using PyTorch. The project is based on the **Oxford-IIIT Pet Dataset**, which provides:

* 37-class breed labels (classification)
* Bounding box annotations (localization)
* Pixel-level trimaps (segmentation)

The objective of this assignment is to design a unified system capable of:

1. Classifying the pet breed
2. Localizing the pet using bounding boxes
3. Segmenting the pet at the pixel level

Unlike solving these tasks independently, this project integrates all three into a **single cohesive deep learning pipeline**.

---

## Assignment Constraints Followed

The implementation strictly follows all assignment requirements:

* VGG11 architecture implemented **from scratch** (no pretrained torchvision models)
* Use of **Batch Normalization** to stabilize training
* Implementation of **Custom Dropout layer** (without using torch.nn.Dropout)
* Strict **train/test separation** to avoid data leakage
* Localization output format:

  ```
  [x_center, y_center, width, height]
  ```

  in pixel coordinates
* Implementation of a **Custom IoU Loss**
* Use of only allowed libraries:

  * torch, numpy, matplotlib, scikit-learn, wandb, albumentations
* Integration of all tasks into a **single forward pass model**
* Logging and experimentation tracked using **Weights & Biases (W&B)**

---

## Project Overview

The pipeline is divided into four main stages:

### 1. Classification (VGG11)

A VGG11-based model is trained to classify pet breeds into 37 classes.

* Batch Normalization improves convergence
* Custom Dropout reduces overfitting
* The model learns hierarchical visual features from edges to semantic patterns

---

### 2. Object Localization

The classification backbone is reused as an encoder, and a regression head is added.

* Output: `[x_center, y_center, width, height]`
* Loss used:

  * Mean Squared Error (MSE)
  * Custom IoU Loss

This improves both coordinate accuracy and spatial overlap quality.

---

### 3. Semantic Segmentation (U-Net)

A U-Net style architecture is built using the VGG11 encoder.

Key design features:

* Transposed convolutions for learnable upsampling
* Skip connections for feature fusion
* Symmetric encoder-decoder structure

Outputs:

* Pixel-wise segmentation mask (trimap)

Evaluation metrics:

* Pixel Accuracy
* Dice Score

---

### 4. Unified Multi-Task Pipeline

All three tasks are combined into a single model:

* Shared encoder (VGG11 backbone)
* Three task-specific heads:

  * Classification head
  * Localization head
  * Segmentation head

A single forward pass produces:

```
(class_logits, bounding_box, segmentation_mask)
```

---

## Weights & Biases (W&B) Experiments

All experiments were tracked and visualized using W&B.

---

### 2.1 Regularization Effect of BatchNorm

Batch Normalization stabilized activation distributions and improved training behavior. It allowed faster convergence, supported higher learning rates, and reduced internal covariate shift. As a result, the model trained more efficiently and reached better performance compared to the version without BatchNorm.

---

### 2.2 Effect of Custom Dropout

Three settings were compared:

* No Dropout
* Dropout (p = 0.2)
* Dropout (p = 0.5)

Without dropout, the model overfitted quickly, showing a large gap between training and validation loss. With dropout, this gap reduced significantly, indicating better generalization. A higher dropout rate provided stronger regularization but slowed down convergence.

---

### 2.3 Transfer Learning Showdown

Three strategies were evaluated:

1. Strict Feature Extractor  
   The encoder was completely frozen and only the decoder was trained. This approach was computationally efficient but resulted in lower performance.

2. Partial Fine-Tuning  
   Early layers were frozen while deeper layers were trained. This provided a balance between performance and efficiency.

3. Full Fine-Tuning  
   The entire network was trained end-to-end. This approach achieved the best final performance.

**Conclusion:**  
Fine-tuning deeper layers improves performance because higher-level features are more task-specific and adaptable to segmentation.

---

### 2.4 Inside the Black Box: Feature Maps

Feature maps from early and late layers were compared using a sample image.

The first convolutional layer captured simple features such as edges, textures, and color contrasts. These features are localized and low-level. In contrast, the last convolutional layer captured higher-level semantic information such as the pet’s face, ears, and body shape.

This demonstrates the hierarchical nature of CNNs, where simple features are gradually combined into more complex and meaningful representations.

---

### 2.5 Object Detection: Confidence & IoU

A W&B table was created containing 10 images with:

* Ground truth bounding boxes (green)
* Predicted bounding boxes (red)
* IoU scores
* Confidence scores

The IoU values varied significantly across examples, indicating differences in localization quality. Higher IoU values corresponded to accurate bounding boxes, while lower IoU values indicated poor overlap.

A failure case was observed where the model produced a relatively high confidence score but a low IoU. This indicates that the classifier was confident about the object inside the predicted crop, even though the bounding box was not well aligned with the ground truth.

This issue was mainly caused by:

* Complex or cluttered backgrounds
* Partial occlusion of the pet
* Small object size relative to the image

These factors made it difficult for the localization model to accurately predict bounding box coordinates.

---

### 2.6 Segmentation Evaluation: Dice vs Pixel Accuracy

Both Pixel Accuracy and Dice Score were tracked during validation.

It was observed that Pixel Accuracy appeared relatively high even in early stages of training, while the Dice Score remained low. This occurs because most pixels in the image belong to the background. If the model predicts background correctly, Pixel Accuracy becomes high even if the object segmentation is poor.

In contrast, the Dice Score measures the overlap between predicted and ground truth regions. Since it focuses on the foreground object, it provides a more meaningful evaluation of segmentation performance.

**Conclusion:**  
Dice Score is a more reliable metric for segmentation tasks with class imbalance, as it penalizes incorrect object predictions more effectively than Pixel Accuracy.

---

### 2.7 Final Pipeline Showcase

The final pipeline was tested on new images outside the dataset.

The model generalized reasonably well:

* Classification worked well for clear images
* Localization was accurate when the object was clearly visible
* Segmentation struggled in challenging conditions such as unusual lighting and complex backgrounds

---

### 2.8 Meta-Analysis and Reflection

This assignment shows how model design and training choices affect multi-task performance.

Batch Normalization made training more stable and faster, while Custom Dropout reduced overfitting and improved generalization. Together, they improved overall performance.

Fine-tuning the encoder gave better results than freezing it, because deeper layers learn task-specific features useful for localization and segmentation. The shared backbone worked well without much task interference.
Loss functions also mattered. IoU loss improved bounding box accuracy, and Dice Score gave a more reliable measure of segmentation performance than Pixel Accuracy.

---

## Repository Structure

```
.
├── checkpoints
├── data
│   └── pets_dataset.py
├── losses
│   └── iou_loss.py
├── models
│   ├── classification.py
│   ├── localization.py
│   ├── segmentation.py
│   ├── multitask.py
│   ├── vgg11.py
│   └── layers.py
├── train.py
├── inference.py
├── requirements.txt
├── Qn_2.4.py
├── Qn2.5.py
├── Qn_2.6.py
├── Qn_2.7.py
└── README.md
```

---

## Conclusion

This project demonstrates how classification, localization, and segmentation can be integrated into a unified deep learning pipeline. The results show that shared representations, proper regularization, and careful evaluation metrics are critical for building effective visual perception systems. The assignment provides practical insight into real-world computer vision pipelines and multi-task learning.
