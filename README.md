# Facial Emotion Detection Using Convolutional Neural Networks and Representational Autoencoder Units

**Group Number**: 14  
**Members**: Rami Amasha, Ahmad Bsese

## Problem Definition
Facial emotion detection involves identifying emotions expressed on faces, important in fields such as psychology, marketing, and human-computer interactions.

## Project Goal
Develop an approach for facial emotion detection using CNNs and Autoencoders.

## Previous Approaches
- **Rule-Based Systems**: Rely on predefined rules or heuristics to identify emotions.
- **SVMs (Support Vector Machines)**: Find the hyperplane in high-dimensional space that maximally separates different classes.

## CNNs & Autoencoders
- **CNN (Convolutional Neural Networks)**: Deep learning models well-suited for image classification, composed of layers that extract features from input data.
- **Autoencoders**: Neural networks used to learn efficient data representations, consisting of an encoder (maps data to a low-dimensional representation) and a decoder (maps back to the original dimension).

## Methodology
### Dataset
**JAFFE Dataset**: 213 images of Japanese females displaying 7 basic emotions (angry, happy, disgust, surprised, fear, neutral, sad).

### Preprocessing
Data augmentation: For each image, 16 batches of size 48x48 were generated.

### CNN Architecture
1. **Input Layer**: Learns features from input data, performing a dot product between filters and a small region of the input data, then applying an activation function.
2. **Pooling Layer**: Reduces the size of feature maps, using methods like max pooling or average pooling.
3. **Fully Connected Layers**: Makes predictions on the learned features, taking the flattened feature maps as input and applying a dot product with weights and an activation function to produce the output.
4. **Output Layer**: Produces the network's output, with the number of nodes matching the number of classes to predict.

### Training Procedures (CNN & Autoencoder)
1. Train the autoencoder to reconstruct input images, using learned weights as initial weights for the classifier network, which is trained to classify images by emotions.
2. Use the encoder part of the autoencoder as the feature extractor for the CNN.

## Results
### CNN & Autoencoder Model on JAFFE Dataset
- **Test Loss**: 0.10
- **Test Accuracy**: 63.65%
- **FER2013 Dataset**: 
  - **Test Loss**: 0.17
  - **Test Accuracy**: 48.9%

### CNN Model on JAFFE Dataset
- **Correct Predictions**: 799 out of 864
- **Accuracy**: 92.4%

### CNN Model on FER2013 Dataset
- **Correct Predictions**: 2942 out of 3589
- **Accuracy**: 81.97%

## Conclusions
- The CNN model performed better than the combined CNN and autoencoder model, possibly due to:
  - Overfitting: The combined model may overfit due to more parameters and irrelevant features learned by the autoencoder.
  - Feature Representation: The CNN may better extract task-specific features directly from the input.
  - Data Quality: The CNN model might benefit from better data quality or quantity.

## Code Files
- **Training the CNN Model**: `CNN_train`
- **Training and Testing the CNN-Autoencoder Model**: `cnn-autoencoder-finalVersion`
- **Testing the CNN Model**: `CNN-TEST-JAFFE_FER2013`

### Note
Please do not delete any file from the directory.
