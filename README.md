# Landmark Detection 🌍📸

## Overview

The Landmark Detection project utilizes deep learning techniques to classify images of landmarks into different categories. It uses a Convolutional Neural Network (CNN) built with Keras to train a model on the landmark dataset, and subsequently, it displays the classified images along with their labels.

## Features 🌟

- **Data Handling:** Load and preprocess the landmark dataset.
- **Data Augmentation:** Preprocess images to fit the model requirements.
- **Model Building:** Create a CNN model using Keras with a VGG19 backbone.
- **Training & Evaluation:** Train the model and evaluate its performance on test data.
- **Visualization:** Display images along with their predicted labels.

## Installation 🛠️

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Landmark-Detection.git
    ```
    
2. **Navigate to the project directory:**
    ```bash
    cd Landmark-Detection
    ```

3. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage 🚀

1. **Run the project:**
    ```bash
    python app.py
    ```

2. **The model will display images from the dataset and predict their classes.**

## Dataset 📚

The dataset used consists of images of landmarks categorized into different classes. The images are stored in a directory structure, and a CSV file provides labels for each image.

## Model Architecture 🏗️

The model is based on a Convolutional Neural Network with the following architecture:

- **Base Model:** VGG19 with weights initialized to `None`.
- **Additional Layers:** Dense layer for classification with dropout layers for regularization.
- **Output:** Softmax activation function for multi-class classification.

## Results 📊

The model is trained and evaluated on the landmark dataset. The performance metrics are reported, and the model is capable of classifying images into various landmark categories with high accuracy.

## Example Output 🖼️

Below are examples of the model's predictions on images from the test set:

- **Image 1:** Class: Landmark A
- **Image 2:** Class: Landmark B

## Acknowledgements 🙏

- The dataset is provided by the respective source.
- TensorFlow and Keras libraries are used for building and training the model.
