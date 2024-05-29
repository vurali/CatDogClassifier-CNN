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

# Initialize the ImageDataGenerator with the specified augmentations
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('dataset/dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# Initialize the ImageDataGenerator for the test set
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('dataset/dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# Build the CNN model
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compile the model
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train the model
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)
```

### Making Predictions
The notebook also includes code to make predictions on new images:
```python
import numpy as np
from keras.api.preprocessing import image

test_image = image.load_img('dataset/dataset/cat_or_dog_2.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)
```














