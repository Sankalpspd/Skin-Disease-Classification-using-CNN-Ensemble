# Skin-Disease-Classification-using-CNN-Ensemble-with-Grad-CAM
This project implements an end-to-end deep learning pipeline for multi-class skin disease classification.

The system uses an ensemble of MobileNetV2, EfficientNet-B1, and ResNet50 architectures with custom classifier heads. Model predictions are combined using weighted probability averaging to improve robustness.

Grad-CAM visualization is integrated to provide interpretability by highlighting image regions influencing the prediction.

The final model is deployed as an interactive web application using Streamlit.

# Dataset

The dataset consists of approximately 35,000 dermoscopic skin lesion images across 6 diagnostic categories. The images represent a diverse set of dermatological conditions captured under varying lighting, resolution, and background conditions.

# Classes

The dataset includes the following six classes:

Enfeksiyonel

Ekzama

Akne

Pigment (Benign Pigmented Lesions)

Pigment (Other Pigmented Conditions)

Malign

# Dataset Challenges

The dataset exhibited significant class imbalance, with certain classes (e.g., common benign conditions) having substantially more samples than rarer categories such as malignant lesions.

Class imbalance can lead to:

Biased predictions toward majority classes

Poor recall for minority classes

Misleading overall accuracy

# Imbalance Handling Strategy

To address this issue, a balanced training dataset was constructed using controlled random sampling.

For each class, a fixed number of samples (e.g., 3000 per class) was selected.

If a class had fewer than the required samples, sampling was performed with replacement.

The balanced dataset was created once and used consistently across all models to ensure identical class-to-index mappings.

This approach:

Prevented majority-class dominance

Stabilized training across architectures

Improved fairness in ensemble predictions


# Training Process

#1. Transfer Learning Strategy

All models were initialized with ImageNet pretrained weights and fine-tuned on the skin lesion dataset.

The following architectures were used:

MobileNetV2

EfficientNet-B1

ResNet50

Final predictions were generated via weighted ensemble averaging

This approach allowed:

Faster convergence

Better feature generalization

Reduced training time compared to training from scratch

# Model Architecture

# 1. Base Architectures

This project leverages three pretrained convolutional neural network architectures:

MobileNetV2 – lightweight and computationally efficient (3.5 million parameters).

EfficientNet-B1 – optimized compound scaling of depth, width, and resolution (7.8 million parameters).

ResNet50 – deep residual network with skip connections for stable gradient flow (25.6 million parameters).

All models were initialized with ImageNet pretrained weights and fine-tuned on the skin disease dataset.

The variation in model size (3.5M to 25M parameters) provided architectural diversity, which contributed to improved ensemble generalization and reduced prediction variance.

# 2. Custom Classification Heads

The original classifier layers were replaced with custom fully connected layers to adapt the networks to the 6-class skin disease classification task.

MobileNetV2 Head:

Dropout (0.3)

Linear → 512

ReLU

Dropout (0.3)

Linear → 256

ReLU

Dropout (0.3)

Linear → 6

EfficientNet-B1 Head:

Dropout (0.3)

Linear → 512

ReLU

Dropout (0.3)

Linear → 6

ResNet50 Head:

Dropout (0.4)

Linear → 1024

BatchNorm

ReLU

Dropout (0.4)

Linear → 512

BatchNorm

ReLU

Dropout (0.3)

Linear → 6

The use of dropout layers reduced overfitting, while deeper fully connected layers allowed better feature adaptation for dermatological classification.

# 3. Ensemble Architecture

Final predictions were generated using weighted probability averaging:

Final Probability =
0.30 × MobileNetV2 + 0.30 × EfficientNet-B1 + 0.40 × ResNet50

# Performance Metrics
1. Evaluation Strategy

The following metrics were used:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

2. Individual Model Performance

MobileNetV2: 68.34% accuracy, 62.03% precision, 68.48% recall, 63.58 F1 score on validation dataset

EfficientNet-B1: 68.32% accuracy, 62.49% precision, 68.42% recall, 64.54 F1 score on validation dataset

ResNet50: 71.27% accuracy, 65.38% precision, 62.29% recall, 67.18 F1 score on validation dataset

While individual performance was comparable, class-wise performance varied depending on lesion type and visual similarity between categories.

3.Ensemble Test Performance

The weighted ensemble achieved:

Accuracy: 72.07%

Weighted F1-score: 72.69%

Macro F1-score: 68.54%

Macro recall: 74.31

Macro precision: 66.36%

The ensemble improved stability across classes compared to individual models and demonstrated strong recall for minority categories.

Confusion Matrix:
    1.   2.   3.   4.   5.   6.
  1.  490  101  74   60   15   10
  2.  89   334  38   41   4    4
  3.  10   1    295  15   1    0
  4.  10   10   10   106  0    0
  5.  114  20   32   43   986  166
  6.  53   21   16   23   116  620

1. Enfeksiyonel, 2. Ekzama, 3. Akne, 4. Pigment, 5. Benign, 6. Malign
 
The confusion matrix reveals moderate inter-class confusion between visually similar categories, particularly among pigmented and inflammatory lesion types. This behavior aligns with known visual overlap in dermatological imaging.

# Explainability with Grad-CAM

To enhance model transparency, Grad-CAM was integrated into the inference pipeline.

Grad-CAM generates class-specific heatmaps by computing gradients of the predicted class with respect to the final convolutional feature maps.

These heatmaps are overlaid on the original image to visualize regions contributing most strongly to the prediction.

Observations:

In most cases, the models focused on the central lesion area.

Minimal activation was observed in irrelevant background regions.

For malignant predictions, activation patterns concentrated around irregular borders and high-contrast structures.

This interpretability component improves trustworthiness and aligns with clinical decision-support requirements.

# Model Deployment

# Deployment Framework

The trained ensemble model was deployed as an interactive web application using Streamlit, enabling real-time inference and visualization.

The application allows users to:

Upload a skin lesion image (JPG/PNG/JPEG)

View the predicted class

See model confidence score

Visualize Grad-CAM heatmaps for explainability

# Deployment Architecture

The deployment pipeline consists of:

Model Loading

Pretrained weights (.pth files) are loaded using load_state_dict

Models are set to evaluation mode using .eval()

Inference runs on GPU if available, otherwise CPU

Preprocessing

Image resized to 224 × 224 and converted to tensor

Normalized using ImageNet mean and standard deviation

# Ensemble Inference

Predictions generated from:

MobileNetV2

EfficientNet-B1

ResNet50

Softmax probabilities computed

Weighted averaging applied:

0.30 × MobileNetV2
0.30 × EfficientNet-B1
0.40 × ResNet50

Final class selected via maximum probability

# Explainability

Grad-CAM applied to the final convolutional layers

Activation maps resized and averaged

Heatmap overlay generated using Matplotlib

Displayed in the web interface
