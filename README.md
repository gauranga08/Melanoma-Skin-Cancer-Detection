# Melanoma Skin Cancer Detection Using CNN

## Objective
This project implements a Convolutional Neural Network (CNN) to classify skin lesion images as benign or malignant (melanoma). By leveraging deep learning, it aims to aid dermatologists in early detection, enhancing diagnostic accuracy and reducing mortality rates associated with melanoma.

## Technology Overview
- **Framework**: TensorFlow and Keras are used to design, train, and evaluate the CNN model.
- **Dataset**:
  - Images are structured in `train/` and `test/` directories, categorized into `benign/` and `malignant/` subfolders.
- **Preprocessing**:
  - Images are resized to a uniform shape (e.g., 224x224).
  - Pixel values are normalized to the range [0, 1].
- **Training Configuration**:
  - Optimizer: Adam
  - Loss Function: Binary cross-entropy
- **Evaluation Metric**:
  - Accuracy is the primary metric used for training and validation.

## Dataset Structure
The dataset is organized into the following structure:
- **train/**: Contains training images categorized into different skin lesion types.
- **test/**: Contains test images categorized into different skin lesion types.

Each folder represents a different type of lesion, with the images inside labeled according to their category.

## Commit History
Here’s a summary of the recent changes in the dataset:
- **Added test and train files** for the following categories:
  - Actinic keratosis
  - Basal cell carcinoma
  - Dermatofibroma
  - Melanoma
  - Nevus
  - Pigmented benign keratosis
  - Seborrheic keratosis
  - Squamous cell carcinoma
  - Vascular lesion

This update was made **5 days ago**.

## Design

### Process Flow

1. **Data Loading**:
   - Images are loaded using TensorFlow's `image_dataset_from_directory` utility.
   - The data is split into training and validation datasets, with a separate test dataset.

2. **Data Preprocessing**:
   - Resized images to the required input dimensions (224x224x3).
   - Pixel values normalized to a [0, 1] range for stability during training.

3. **Model Architecture**:
   - **Convolutional Layers**: Extract meaningful features from images.
   - **Pooling Layers**: Reduce spatial dimensions to simplify learning.
   - **Dropout Layers**: Prevent overfitting.
   - **Dense Layers**: Perform binary classification using the sigmoid activation function.

4. **Training**:
   - The model is trained with early stopping to prevent overfitting.
   - The validation loss is monitored to determine when training stops.

5. **Evaluation**:
   - Accuracy is calculated on the test dataset to measure the model's performance.

## Key Features

### Model Architecture
- **Input shape**: 224x224x3 (RGB images).
- **Convolutional layers**: Learn spatial features.
- **Max pooling**: Reduce spatial dimensionality.
- **Dropout**: Regularization to prevent overfitting.
- **Dense layer**: Final layer with sigmoid activation for binary classification.

### Preprocessing
- **Resizing** to 224x224.
- **Normalizing** pixel values to the range [0, 1].

### Training Process
- **Loss Function**: Binary cross-entropy.
- **Optimizer**: Adam.
- **Early Stopping**: Stops training when validation loss stops improving.

## Setup Instructions

### Prerequisites
- Python 3.8 or higher.
- TensorFlow and Keras installed.

### Clone the Repository
To get started, clone the repository to your local machine:
```bash
git clone https://github.com/gauranga08/Melanoma-Skin-Cancer-Detection.git
cd Melanoma-Skin-Cancer-Detection

### Create a Virtual Environment
It is recommended to create a virtual environment to manage dependencies. Run the following command:

bash
Copy code
python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
### Install Dependencies
Install the required Python libraries using pip:

bash
Copy code
pip install -r requirements.txt
### Dataset Setup
Ensure your dataset follows the structure outlined in the Dataset Structure section. You should place the images in the appropriate train/ and test/ directories.

### Run the Notebook
You can now run the Jupyter notebook to train and evaluate the model:

bash
Copy code
jupyter notebook melanoma_skin_detection.ipynb
This will open the notebook in your default web browser where you can interact with the code and see the results of training.

### Outputs
- Training Logs: Training and validation accuracy and loss per epoch.
- Final Test Accuracy: Evaluated on the test dataset.
- Saved Model: The trained model is saved locally for future predictions.
### Scope for Improvement
Missing Metrics
Precision, Recall, and F1 Score: Add these metrics to better evaluate the model's performance, especially for unbalanced datasets.
ROC-AUC: Implement ROC-AUC and plot the ROC curve for detailed classification evaluation.
Visualization
Grad-CAM: Visualize which parts of an image the model focuses on for predictions, enhancing interpretability.
Confusion Matrix: Display the confusion matrix to better understand model performance, including false positives and false negatives.
Advanced Techniques
Transfer Learning: Use pre-trained models like ResNet, InceptionNet, or EfficientNet to improve accuracy and reduce training time.
Augmentation: Apply random flips, rotations, and zooming to improve generalization.
Hyperparameter Tuning: Optimize learning rates, batch sizes, and other hyperparameters for better performance.
Cross-Validation: Use K-fold cross-validation to validate the model on multiple splits of the dataset.
Explainability: Leverage tools like SHAP or LIME to explain individual predictions.
Deployment
Package the model into a web application using Flask, FastAPI, or Streamlit for real-world usability.
Acknowledgments
This project is inspired by advancements in AI for medical imaging. It aims to provide an accessible tool for melanoma detection while highlighting areas for future research and development.

