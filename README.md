
# Emotion Detection Using Deep Learning

This project focuses on detecting human emotions from facial images using Convolutional Neural Networks (CNNs).

# Data Set
https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset/data


- **Format**: 48x48 pixel grayscale images
- **Classes**:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Sad
  - Surprise
  - Neutral

The dataset used to predict the emotion is gathered from kaggle

## 🧠 Model Architecture

The model is built using TensorFlow/Keras and consists of:

- Convolutional layers with ReLU activation
- MaxPooling layers for dimensionality reduction
- Dropout for regularization
- Dense (fully connected) layers
- Softmax activation in the output layer for classification

> Optionally, models like **VGG16** or **ResNet50** can be used for transfer learning.

## ⚙️ Dependencies

```bash
tensorflow
numpy
matplotlib
pandas
scikit-learn
opencv-python
