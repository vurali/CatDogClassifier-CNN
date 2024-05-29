# Cat and Dog Image Classifier

## Overview
This project uses a Convolutional Neural Network (CNN) to classify images of cats and dogs. The model is built using TensorFlow and Keras and is trained on a dataset of cat and dog images. The repository includes scripts for data preprocessing, model training, and prediction.

## Project Structure
- `dataset/`: Contains the training and test datasets structured in subdirectories by class.
  - `dataset/training_set/`: Training images.
  - `dataset/test_set/`: Test images.
- `README.md`: Project documentation.
- `requirements.txt`: List of dependencies.
- `train_model.py`: Script to preprocess data, build and train the model.
- `predict.py`: Script to make predictions on new images.

## Installation
Clone the repository and install the required packages:
```bash
git clone https://github.com/vurali/CatDogClassifier-CNN.git
cd CatDogClassifier-CNN
pip install -r requirements.txt
