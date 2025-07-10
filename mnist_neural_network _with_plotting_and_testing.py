import numpy as np
import matplotlib.pyplot as plt
import csv
import random
from collections import defaultdict
import pickle
import os

"""This following code is for training a neural network to classify MNIST digits.
It includes functions for loading data, preprocessing, training the model, evaluating performance, and visualizing results.
The neural network consists of an input layer, multiple hidden layers with ReLU activation, and an output layer with softmax activation.
It also includes functions for plotting training history and visualizing predictions.

The naming convention for the model is as follows:
X: Input dataset
y: Output dataset
X_train: Training data
y_train: Training labels
X_val: Validation data
y_val: Validation labels

For other variable names as well, I have tried to use variable use standardly used in the machine learning and neural network's community
"""

class NeuralNetwork:
    def __init__(self, input_size=784, hidden_sizes=[128, 64], output_size=10, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layer_sizes) - 1):
            weight = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            bias = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(weight)
            self.biases.append(bias)

        # Lists to store training and validation losses and accuracies for plotting later     
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward_propagation(self, X):
        activations = [X]
        z_values = []
        current_input = X
        for i in range(len(self.weights) - 1):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            z_values.append(z)    
            activation = self.relu(z)
            activations.append(activation)
            current_input = activation     
        z_output = np.dot(current_input, self.weights[-1]) + self.biases[-1]
        z_values.append(z_output)
        output = self.softmax(z_output)
        activations.append(output)
        return activations, z_values
    
    def cross_entropy_loss(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss=-np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return loss
    
    def backward_propagation(self, X, y_true, activations, z_values):
        batch_size = X.shape[0]
        weight_gradients = []
        bias_gradients = []
        for i in range(len(self.weights)):
            weight_gradients.append(np.zeros_like(self.weights[i]))
            bias_gradients.append(np.zeros_like(self.biases[i]))
        delta = activations[-1] - y_true
        for i in range(len(self.weights) - 1, -1, -1):
            weight_gradients[i] = np.dot(activations[i].T, delta) / batch_size
            bias_gradients[i] = np.mean(delta, axis=0, keepdims=True)
            if i > 0:  # Not the input layer
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(z_values[i-1])
        return weight_gradients, bias_gradients
    
    def update_parameters(self, weight_gradients, bias_gradients):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32, verbose=True):
        n_samples = X_train.shape[0]
        n_batches = n_samples // batch_size
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_accuracy = 0
            indices = np.random.permutation(n_samples)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, n_samples)
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]
                
                # Forward propagation using RELU activation
                activations, z_values = self.forward_propagation(X_batch)
                
                # Calculating loss using cross-entropy
                loss = self.cross_entropy_loss(y_batch, activations[-1])
                epoch_loss += loss
                
                # Calculating accuracy
                predictions = np.argmax(activations[-1], axis=1)
                true_labels = np.argmax(y_batch, axis=1)
                accuracy = np.mean(predictions == true_labels)
                epoch_accuracy += accuracy
                
                # Backward propagation
                weight_gradients, bias_gradients = self.backward_propagation(
                    X_batch, y_batch, activations, z_values
                )
                
                # Updating parameters
                self.update_parameters(weight_gradients, bias_gradients)
            
            avg_loss = epoch_loss / n_batches
            avg_accuracy = epoch_accuracy / n_batches
            self.train_losses.append(avg_loss)
            self.train_accuracies.append(avg_accuracy)
            
            if X_val is not None and y_val is not None:
                val_loss, val_accuracy = self.evaluate(X_val, y_val)
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_accuracy)
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Loss: {avg_loss}, Accuracy: {avg_accuracy}, "
                          f"Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}:"
                          f"Loss: {avg_loss}, Accuracy: {avg_accuracy}")
    
    def evaluate(self, X, y):
        activations, _ = self.forward_propagation(X)
        loss = self.cross_entropy_loss(y, activations[-1])
        predictions = np.argmax(activations[-1], axis=1)
        true_labels = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == true_labels)   
        return loss, accuracy
    
    def predict(self, X):
        activations, _ = self.forward_propagation(X)
        return np.argmax(activations[-1], axis=1)
    
    def predict_probability(self, X):
        activations, _ = self.forward_propagation(X)
        return activations[-1]

def load_data(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        header = next(reader) 
        for row in reader:
            data.append([float(x) for x in row])
    return np.array(data)

def classify_data(data, has_labels=True):
    if has_labels:
        X = data[:, 1:]  
        y = data[:, 0].astype(int)     
        X = X / 255.0
        y_one_hot = np.zeros((len(y), 10))
        y_one_hot[np.arange(len(y)), y] = 1
        return X, y_one_hot, y
    else:
        X = data / 255.0
        return X

def split_validationdata(X, y, y_original, validation_ratio=0.2):
    n_samples = X.shape[0]
    n_val = int(n_samples * validation_ratio)
    indices = np.random.permutation(n_samples)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    X_train = X[train_indices]
    y_train = y[train_indices]
    y_train_original = y_original[train_indices]
    X_val = X[val_indices]
    y_val = y[val_indices]
    y_val_original = y_original[val_indices]
    return X_train, y_train, y_train_original, X_val, y_val, y_val_original

def plot_training_history(model):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(model.train_losses, label='Training Loss', color='blue')
    if model.val_losses:
        ax1.plot(model.val_losses, label='Validation Loss', color='red')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(model.train_accuracies, label='Training Accuracy', color='blue')
    if model.val_accuracies:
        ax2.plot(model.val_accuracies, label='Validation Accuracy', color='red')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def visualize_predictions(X, y_true, y_pred, num_samples=10):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    indices = np.random.choice(len(X), num_samples, replace=False)
    for i, idx in enumerate(indices):
        image = X[idx].reshape(28, 28)
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f'True: {y_true[idx]}, Pred: {y_pred[idx]}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

def test_model(model, test_filename="test.csv", output_filename="predictions.csv"):
    print("Loading test data from test.csv")
    test_data = load_data(test_filename)
    X_test = classify_data(test_data, has_labels=False)
    print(f"Making predictions on {len(X_test)} samples...")
    predictions = model.predict(X_test)
    visualize_predictions(X_test, np.zeros(len(predictions)), predictions, num_samples=min(100, len(predictions)))
    if os.path.exists(output_filename):
        try:
            os.remove(output_filename)
            print(f"File '{output_filename}' deleted successfully.")
        except OSError as e:
            print(f"Error deleting file '{output_filename}': {e}")
    with open(output_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['id', 'label'])  # Header    
            for i, pred in enumerate(predictions):
                writer.writerow([i+1, pred]) # To start indexing from 1 
    print(f"Predictions saved to {output_filename}")
    return predictions

def main():
    print("Starting the program")
    print("Loading training data...")
    try:
        training_data = load_data("train.csv") 
        X, y_one_hot, y_original = classify_data(training_data, has_labels=True)
        print(f"Training data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    except FileNotFoundError:
        print("Error: train.csv not found. Please ensure the file is in the current directory.")
        return
    
    # Split data
    print("Splitting data into training and validation sets...")
    X_train, y_train, y_train_orig, X_val, y_val, y_val_orig = split_validationdata(
        X, y_one_hot, y_original, validation_ratio=0.2
    )   
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print("\nBuilding the neural network:")

    hiddens_Layers=[]
    while True:
        input_size=input("Enter hidden layer size: ")
        if input_size=='':
            if len(hiddens_Layers) == 0:
                print("No hidden layer size entered. At least one hidden layer is required.")
                continue
            else:
                print("No more hidden layers will be added.")
                break
        else:
            if type(eval((input_size))) is not int or eval(input_size) <= 0:
                print("Please enter a valid positive integer for the hidden layer size.")
                continue
            hiddens_Layers.append(int(input_size))
    while True:
        learning_rate_input = input("Enter learning rate (default is 0.01): ")
        if learning_rate_input == '':
            learning_rate = 0.01
            break
        
        if type(eval(learning_rate_input)) is not float or type(eval(learning_rate_input)) is not int:
            print("Please enter a valid number for the learning rate.")
            continue
        else:
            if eval(learning_rate_input) < 0:
                print("Learning rate cannot be negative. Please enter a positive value.")
                continue
            if eval(learning_rate_input) == 0:
                print("Learning rate cannot be zero. Please enter a positive value.")
                continue
            learning_rate = float(learning_rate_input)
        
        break
    
        
    model = NeuralNetwork(
        input_size=784,
        hidden_sizes=hiddens_Layers, 
        output_size=10,
        learning_rate=learning_rate
    )
    
    print("Model structure:")
    print(f"Input Layer: {model.input_size} neurons")
    for i, size in enumerate(model.hidden_sizes):
        print(f"Hidden Layer {i+1}: {size} neurons (ReLU)")
    print(f"Output Layer: {model.output_size} neurons (Softmax)")
    
    print("\nTraining the model:")
    model.train(
        X_train, y_train, 
        X_val, y_val, 
        epochs=100, 
        batch_size=32, 
        verbose=True
    )
    
    print("\nPlotting training history...")
    plot_training_history(model)
    
    print("\nFinal Model Evaluation:")
    training_loss, training_accuracy = model.evaluate(X_train, y_train)
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print(f"Training - Loss: {training_loss}, Accuracy: {training_accuracy}")
    print(f"Validation - Loss: {val_loss}, Accuracy: {val_accuracy}")

    print("\nVisualizing predictions")
    samples=input("Enter number of samples to visualize predictions (default is 10): ") 
    if samples == '':
        samples = 10
    elif type(eval((samples))) is not int or eval(samples) <= 0:
        print("Entered value is not a valid positive integer. Defaulting to 10 samples.")
        samples=10
    samples = int(samples)
    if samples > 100:
        print("Limiting the number of samples to 100 for visualization.")
        samples = 100
    if samples > len(X_val):
        print(f"Only {len(X_val)} samples available for visualization. Adjusting to {len(X_val)} samples.")
        samples = len(X_val)
    val_predictions = model.predict(X_val)
    visualize_predictions(X_val, y_val_orig, val_predictions, num_samples=samples)
    
    print("\nTesting on test data...")
    try:
        test_predictions = test_model(model, 'test.csv', 'predictions.csv')
        print(f"Test predictions completed!")
    except FileNotFoundError:
        print("test.csv not found. Skipping test predictions.")

    # Save the trained model for future use
    # Please ensure that you have saved the previous model elsewhere or you are okay with overwriting it.
    print("\nSaving the trained model to 'trained_model.pkl'")

    if os.path.exists('trained_model.pkl'):
        try:
            os.remove('trained_model.pkl')
            print("Previous model file deleted successfully.")
        except OSError as e:
            print(f"Error deleting previous model file: {e}")
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(model, f)                     
    print("\n Training completed successfully!")
    return model
np.random.seed(42)
random.seed(42)
trained_model = main()
