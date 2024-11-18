# Melanoma Skin Cancer Detection Using CNN

## Objective
This project implements a Convolutional Neural Network (CNN) to classify skin lesion images as benign or malignant (melanoma). By leveraging deep learning, it aims to aid dermatologists in early detection, enhancing diagnostic accuracy and reducing mortality rates associated with melanoma.

## Technology Overview
- **Framework**: TensorFlow and Keras are used to design, train, and evaluate the CNN model.
- **Dataset**:
  - Images are structured in `Train/` and `Test/` directories, categorized into `benign/` and `malignant/` subfolders.
- **Preprocessing**:
  - Images are resized to a uniform shape (e.g., 180x180).
  - Resizing, normalization, and augmentation techniques (rotation, flipping, zooming) improve robustness.
- **Training Configuration**:
  - Optimizer: Adam
  - Loss Function: Binary cross-entropy
- **Evaluation Metric**:
  - Accuracy is the primary metric used for training and validation.

## Dataset Structure
Here’s a summary of the the dataset:
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

## Design

### Process Flow

1. **Data Loading and Preprocessing**:
   - Images are loaded using TensorFlow's `image_dataset_from_directory` utility.
   - The data is split into training and validation datasets, with a separate test dataset.

2. **Model Architecture**:
   - **Convolutional Layers**: Extract meaningful features from images.
   - **Pooling Layers**: Reduce spatial dimensions to simplify learning.
   - **Dropout Layers**: Prevent overfitting.
   - **Dense Layers**: Perform binary classification using the sigmoid activation function.

4. **Training**:
   - The model is trained with early stopping to prevent overfitting.
   - The validation loss is monitored to determine when training stops.

5. **Evaluation**:
   - Accuracy is calculated

## Setup Instructions

### Prerequisites
- Python 3.8 or higher.
- TensorFlow and Keras installed.

### Clone the Repository
1. To get started, clone the repository to your local machine:
    ```bash
    git clone https://github.com/gauranga08/Melanoma-Skin-Cancer-Detection.git
    cd Melanoma-Skin-Cancer-Detection

### Create a Virtual Environment
1. It is recommended to create a virtual environment to manage dependencies. Run the following command:
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
### Install Dependencies
1. Install the required Python libraries using pip:
  
    ```bash
    pip install -r requirements.txt
### Dataset Setup
Ensure your dataset follows the structure outlined in the Dataset Structure section. You should place the images in the appropriate Train/ and Test/ directories.

### Run the Notebook
You can now run the Jupyter notebook to train and evaluate the model
## Scope for Improvement:

### Advanced Techniques
**Transfer Learning**: Use pre-trained models like ResNet, InceptionNet, or EfficientNet to improve accuracy and reduce training time.
**Augmentation**: Apply random flips, rotations, and zooming to improve generalization.
**Hyperparameter Tuning**: Optimize learning rates, batch sizes, and other hyperparameters for better performance.
**Cross-Validation**: Use K-fold cross-validation to validate the model on multiple splits of the dataset.
**Explainability**: Leverage tools like SHAP or LIME to explain individual predictions.

**Precision, Recall, and F1 Score**: Add these metrics to better evaluate the model's performance, especially for unbalanced datasets.
**ROC-AUC**: Implement ROC-AUC and plot the ROC curve for detailed classification evaluation.
### Visualization
**Grad-CAM**: Visualize which parts of an image the model focuses on for predictions, enhancing interpretability.
**Confusion Matrix**: Display the confusion matrix to better understand model performance, including false positives and false negatives.
### Deployment
Package the model into a web application using Flask, FastAPI, or Streamlit for real-world usability.
## Acknowledgments
This project draws inspiration from various public datasets and deep learning practices aimed at improving medical diagnostics.

Feel free to contribute or suggest enhancements via pull requests! 😊
