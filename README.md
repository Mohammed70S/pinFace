Face Recognition Using VGG16 and Keras
This project implements a face recognition model using the VGG16 architecture as a base, trained on the Pins Face Dataset. The model aims for a classification accuracy of at least 85% on the validation set.

Table of Contents
Project Overview
Installation
Usage
Model Architecture
Data Augmentation
Training
Evaluation
Results
License
Project Overview
The objective of this project is to classify images of faces into predefined categories using a Convolutional Neural Network (CNN) model built with Keras. Transfer learning with the VGG16 architecture is utilized to leverage pre-trained weights for better performance.

Installation
To run this project, you need to have Python installed along with the following packages:

TensorFlow
NumPy
Matplotlib
Kaggle Hub
You can install the required libraries using pip:

bash

pip install tensorflow numpy matplotlib kagglehub
Usage
Download the Dataset: The dataset can be downloaded directly using the Kaggle Hub.

python

import kagglehub

path = kagglehub.dataset_download("hereisburak/pins-face-recognition")
print("Path to dataset files:", path)
Run the Model: Execute the Python script to train the model and evaluate its performance.

Model Architecture
The model architecture consists of:

Base Model: VGG16 (with pre-trained weights from ImageNet)
Flatten Layer: Converts the 2D matrix to a 1D vector.
Dense Layers: Fully connected layers with ReLU activation.
Batch Normalization: Normalizes the output of the previous layer.
Dropout Layers: Regularization technique to prevent overfitting.
Output Layer: Softmax activation for multi-class classification.
python

model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])
Data Augmentation
Data augmentation techniques are applied to enhance the diversity of the training dataset. These include:

Rescaling pixel values
Horizontal flipping
Rotation
Zoom
Brightness adjustment
Shearing
Training
The model is trained using:

Loss Function: Categorical Crossentropy
Optimizer: Adam with a learning rate of 0.001
Callbacks: Early stopping and learning rate reduction on plateaus
python

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=50,
    callbacks=[early_stopping, lr_scheduler]
)
Evaluation
After training, the model is evaluated on the validation set. The validation accuracy is printed, along with a classification report that includes precision, recall, and F1-score.

python

val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {val_accuracy:.2f}")
Results
The training and validation accuracy and loss are plotted to visualize the model's performance over epochs. The target accuracy of 85% is checked, and appropriate messages are printed based on the results.

License
This project is licensed under the MIT License. See the LICENSE file for details.
