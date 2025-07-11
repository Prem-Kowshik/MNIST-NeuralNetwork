# MNIST Neural Network Classifier

A comprehensive implementation of a feedforward neural network for handwritten digit classification using the MNIST dataset. This project demonstrates fundamental deep learning concepts with clear explanations of the technical choices made throughout the implementation.

## 🎯 Project Overview

This project implements a multi-layer neural network from scratch to classify handwritten digits (0-9) from the given dataset. The implementation includes training, validation, testing, and visualization capabilities, making it an excellent learning resource for understanding the fundamentals of neural networks.

### Key Features

- **Pure NumPy Implementation**: Built from scratch without high-level ML frameworks
- **Interactive Training Interface**: Customizable network architecture and hyperparameters
- **Comprehensive Visualization**: Training curves, prediction samples, and performance metrics
- **Robust Data Handling**: Automatic train/validation splitting and data preprocessing
- **Model Persistence**: Save and load trained models for future use
- **Extensive Documentation**: Clear explanations of technical choices and implementations

## 🧠 Technical Architecture

### Network Structure

The neural network follows a classic feedforward architecture:

```
Input Layer (784 neurons) → Hidden Layers (customizable) → Output Layer (10 neurons)
```

- **Input Layer**: 784 neurons (28×28 pixel images flattened)
- **Hidden Layers**: User-configurable sizes with ReLU activation
- **Output Layer**: 10 neurons (one for each digit 0-9) with softmax activation

### Why These Technical Choices?

#### ReLU Activation Function

We use **Rectified Linear Unit (ReLU)** for hidden layers because:

1. **Vanishing Gradient Solution**: Unlike sigmoid or tanh functions, ReLU maintains gradients for positive inputs, preventing the vanishing gradient problem that plagues deep networks

2. **Computational Efficiency**: ReLU is extremely simple to compute (`f(x) = max(0, x)`) compared to exponential functions like sigmoid, making training significantly faster

3. **Sparse Activation**: ReLU naturally creates sparse representations by setting negative values to zero, which can improve model efficiency and reduce overfitting

4. **Better Convergence**: Networks with ReLU typically converge faster during training compared to traditional activation functions[4][7]

The mathematical simplicity of ReLU's derivative (either 0 or 1) makes backpropagation more stable and efficient.

#### Softmax Activation Function

The output layer uses **softmax activation** for several critical reasons:

1. **Probability Distribution**: Softmax converts raw output scores (logits) into a valid probability distribution where all outputs sum to 1, making them interpretable as class probabilities

2. **Multi-class Classification**: Unlike sigmoid (binary), softmax naturally handles multiple classes by computing relative probabilities across all possible outcomes

3. **Mathematical Stability**: The softmax function handles both positive and negative inputs gracefully, unlike simple normalization which fails when the sum of logits is zero

4. **Amplified Differences**: Softmax emphasizes differences between logits through exponentiation, making the model more confident in its predictions

The softmax function is mathematically defined as:
```
softmax(zi) = e^zi / Σ(e^zj) for j=1 to K
```
where K is the number of classes.

#### Cross-Entropy Loss Function

We use **cross-entropy loss** as our optimization objective because:

1. **Probabilistic Foundation**: Cross-entropy naturally measures the difference between the true probability distribution (one-hot encoded labels) and predicted probabilities

2. **Gradient Properties**: When combined with softmax, cross-entropy provides clean, stable gradients for backpropagation, leading to efficient optimization

3. **Classification Optimization**: Cross-entropy is specifically designed for classification tasks and penalizes wrong predictions more heavily than being "close but not perfect"

4. **Information Theory**: It measures the average number of bits needed to encode the true distribution using the predicted distribution, providing a principled approach to model evaluation

The mathematical relationship between softmax and cross-entropy creates elegant gradients that simplify to `predicted_probability - true_label`, making optimization very efficient.

## 📊 Dataset Information

### MNIST Dataset

The dataset was provided by DevSoc and is available at this link: https://drive.google.com/drive/folders/1Tbyf4x_rX8kyQtroqyIpGuaRAHUO5M8_?usp=sharing
- **Size**: 42,000 training images + 28,000 test images
- **Format**: csv files with 785 columns (label+ flattened 28X28 grayscale images of handwritten digits)- train.csv and 784 columns (flattened 28X28 grayscale images of handwritten digits)- test.csv
- **Classes**: 10 digits (0-9)
- **Preprocessing**: Normalized pixel values (0-1) and one-hot encoded labels
- **Origin**: Provided by DevSoc

### Data Pipeline

1. **Loading**: CSV format with first column as labels, remaining 784 columns as pixel values
2. **Normalization**: Pixel values divided by 255 to scale to [0,1] range
3. **Label Encoding**: Convert integer labels to one-hot vectors for training
4. **Train/Validation Split**: 80/20 split with random shuffling

## 🚀 Getting Started

### Prerequisites

```python
numpy
matplotlib
csv
pickle
os
random
collections
```

### Installation & Usage

1. **Clone or download** the neural network script
2. **Prepare your data**: Ensure `train.csv` and `test.csv` are in the same directory (you can download them from the drive link: https://drive.google.com/drive/folders/1Tbyf4x_rX8kyQtroqyIpGuaRAHUO5M8_?usp=sharing)
3. **Run the program**:
   ```bash
   python mnist_neural_network.py
   ```
   ```command prompt
   python mnist_neural_network.py
   ```

### Interactive Configuration

The program will prompt you to configure:

- **Hidden layer sizes**: Enter positive integers (press Enter when done)
- **Learning rate**: Float value (default: 0.01)
- **Visualization samples**: Number of predictions to display (default: 10)

### Example Session

```
Enter hidden layer size: 128
Enter hidden layer size: 64
Enter hidden layer size: [Enter to finish]
Enter learning rate (default is 0.01): 0.001

Model structure:
Input Layer: 784 neurons
Hidden Layer 1: 128 neurons (ReLU)
Hidden Layer 2: 64 neurons (ReLU)
Output Layer: 10 neurons (Softmax)
```

## 📈 Training Process

### Forward Propagation

1. **Input Processing**: Read and process the data from the csv files
2. **Hidden Layer Processing**: Apply linear transformation followed by ReLU activation
3. **Output Generation**: Final linear layer followed by softmax for probability distribution

### Backpropagation

1. **Loss Calculation**: Cross-entropy between predicted and true distributions
2. **Gradient Computation**: Chain rule application through all layers
3. **Weight Updates**: Gradient descent with configurable learning rate

### Training Features

- **Batch Processing**: Configurable batch size (default: 32)
- **Progress Monitoring**: Real-time loss and accuracy tracking
- **Validation**: Separate validation set for unbiased performance estimation
- **Early Insights**: Periodic progress reports every 10 epochs

## 📊 Model Evaluation

### Metrics Tracked

- **Training Loss & Accuracy**: Monitor learning progress
- **Validation Loss & Accuracy**: Detect overfitting
- **Learning Curves**: Visualize training dynamics
- **Prediction Samples**: Visual inspection of model decisions

### Visualization Features

- **Training History Plots**: Loss and accuracy curves over epochs
- **Prediction Visualization**: Grid display of test images with predictions
- **Performance Metrics**: Final evaluation on test set

## 💾 Model Persistence

The trained model is automatically saved as `trained_model.pkl` and includes:

- Network architecture (weights and biases)
- Training history (losses and accuracies)
- Model hyperparameters

Load saved models for inference:
```python
import pickle
with open('trained_model.pkl', 'rb') as f:
    model = pickle.load(f)
```

## 🎯 Expected Performance

With proper configuration, you should expect:

- **Training Accuracy**: 100%+ after 100 epochs
- **Validation Accuracy**: 97%+ (depends on architecture)
- **Convergence**: Steady improvement in first 20-30 epochs
- **Overfitting Signs**: Gap between training and validation metrics

## 🔧 Customization Options

### Network Architecture
- Modify hidden layer sizes for different complexities
- Experiment with different numbers of hidden layers
- Adjust the learning rate for convergence speed

### Training Parameters
- Change batch size in the `train()` method
- Modify epochs for longer/shorter training
- Implement learning rate scheduling


## 📚 Learning Resources

To understand the concepts better:

1. DevSoc Resources
2. sentdex Youtube channel
3. Andrej Karpathy Youtube channel
4. Perplexity and Claude (For error handling and variable names)
