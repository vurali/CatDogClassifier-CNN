# Cat and Dog Image Classifier

## Overview
This project uses a Convolutional Neural Network (CNN) to classify images of cats and dogs. The model is built using TensorFlow and Keras and is trained on a dataset of cat and dog images. The repository includes a Jupyter notebook for data preprocessing, model training, and prediction.

## Project Structure
- `dataset/`: Contains the training and test datasets structured in subdirectories by class.
  - `dataset/training_set/`: Training images.
  - `dataset/test_set/`: Test images.
- `Cat_Dog_Classifier.ipynb`: Jupyter notebook for training and prediction.
- `README.md`: Project documentation.
- `requirements.txt`: List of dependencies.

## Installation
Clone the repository and install the required packages:
```bash
git clone https://github.com/vurali/CatDogClassifier-CNN.git
cd CatDogClassifier-CNN
pip install -r requirements.txt.
```

## Usage
### Running the Jupyter Notebook
- Open the Jupyter notebook:
  ```bash
  jupyter notebook Cat_Dog_Classifier.ipynb
  ```

- Follow the steps in the notebook to:
    - Preprocess the data.
    - Build and train the CNN model.
    - Make predictions on new images.
 

## Example Code Snippets
### Data Preprocessing and Model Training
The notebook includes the following code for data preprocessing and model training:
```python
import tensorflow as tf
from keras.api.preprocessing.image import ImageDataGenerator

# Initialize the ImageDataGenerator for training with specified augmentations
train_datagen = ImageDataGenerator(
    rescale=1./255,        # Rescale pixel values to [0, 1]
    shear_range=0.2,       # Apply random shear transformations
    zoom_range=0.2,        # Apply random zoom transformations
    horizontal_flip=True   # Randomly flip images horizontally
)

# Create an iterator for the training set
training_set = train_datagen.flow_from_directory(
    'dataset/dataset/training_set', # Path to the training set directory
    target_size=(64, 64),           # Resize images to 64x64 pixels
    batch_size=32,                  # Number of images to return in each batch
    class_mode='binary'             # Binary classification mode (e.g., cats vs dogs)
)

# Initialize the ImageDataGenerator for the test set with rescaling only
test_datagen = ImageDataGenerator(rescale=1./255) # Rescale pixel values to [0, 1]

# Create an iterator for the test set
test_set = test_datagen.flow_from_directory(
    'dataset/dataset/test_set',    # Path to the test set directory
    target_size=(64, 64),          # Resize images to 64x64 pixels
    batch_size=32,                 # Number of images to return in each batch
    class_mode='binary'            # Binary classification mode (e.g., cats vs dogs)
)

# Build the Convolutional Neural Network (CNN) model
cnn = tf.keras.models.Sequential()

# Add a convolutional layer with 32 filters, 3x3 kernel size, ReLU activation, and input shape 64x64x3
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

# Add a max pooling layer with 2x2 pool size and stride of 2
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Add another convolutional layer with 32 filters, 3x3 kernel size, and ReLU activation
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))

# Add another max pooling layer with 2x2 pool size and stride of 2
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Flatten the input
cnn.add(tf.keras.layers.Flatten())

# Add a fully connected (dense) layer with 128 units and ReLU activation
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Add the output layer with 1 unit and sigmoid activation for binary classification
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compile the CNN model
cnn.compile(
    optimizer='adam',                # Use Adam optimizer
    loss='binary_crossentropy',      # Use binary cross-entropy loss function
    metrics=['accuracy']             # Evaluate model performance using accuracy
)

# Train the CNN model
cnn.fit(
    x=training_set,                 # Training data
    validation_data=test_set,       # Validation data
    epochs=25                       # Number of epochs to train the model
)

```

### Making Predictions
The notebook also includes code to make predictions on new images:
```python
import numpy as np
from keras.api.preprocessing import image

# Load and preprocess the test image
test_image = image.load_img(
    'dataset/dataset/cat_or_dog_2.jpg', # Path to the image file
    target_size=(64, 64)                # Resize image to 64x64 pixels to match model input
)
test_image = image.img_to_array(test_image) # Convert image to array format
test_image = np.expand_dims(test_image, axis=0) # Add batch dimension since model expects a batch of images

# Predict the class of the image using the trained CNN model
result = cnn.predict(test_image)

# Retrieve the class indices from the training set
class_indices = training_set.class_indices

# Interpret the prediction result
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

# Print the prediction
print(prediction)
```

## Results

The model achieved the following performance metrics on the test set:

- Accuracy: 85%


## Contact
Project Link: https://github.com/vurali/CatDogClassifier-CNN.git

For any inquiries or feedback, please contact:

- Name: Murali Krishna
- Email: pulimurali07@gmail.com












