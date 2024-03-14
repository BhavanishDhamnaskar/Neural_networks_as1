import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class WineQualityClassifier:
    def __init__(self, input_layers, hidden_layers, output_layers, regularization_rate=0.01, dropout_probability=0.5):
        self.layer1_weights = np.random.randn(input_layers, hidden_layers) * 0.01
        self.layer1_bias = np.zeros((1, hidden_layers))
        self.layer2_weights = np.random.randn(hidden_layers, output_layers) * 0.01
        self.layer2_bias = np.zeros((1, output_layers))
        self.reg_rate = regularization_rate
        self.dropout_prob = dropout_probability
        
    def relu_activation(self, inputs):
        return np.maximum(0, inputs)
    
    def relu_derivative(self, derivative_inputs, inputs):
        derivative_outputs = np.array(derivative_inputs, copy=True)
        derivative_outputs[inputs <= 0] = 0
        return derivative_outputs
    
    def softmax_function(self, inputs):
        exp_inputs = np.exp(inputs - np.max(inputs))
        return exp_inputs / exp_inputs.sum(axis=1, keepdims=True)
    
    def calculate_loss(self, true_labels, predicted_labels):
        num_samples = true_labels.shape[0]
        log_probs = -np.log(predicted_labels[range(num_samples), true_labels])
        data_loss = np.sum(log_probs) / num_samples
        reg_loss = (self.reg_rate / (2 * num_samples)) * (np.sum(np.square(self.layer1_weights)) + np.sum(np.square(self.layer2_weights)))
        return data_loss + reg_loss
    
    def forward_pass(self, inputs, training_mode=True):
        layer1_z = np.dot(inputs, self.layer1_weights) + self.layer1_bias
        layer1_a = self.relu_activation(layer1_z)
        if training_mode:
            dropout_mask = np.random.rand(*layer1_a.shape) > self.dropout_prob
            layer1_a *= dropout_mask
            layer1_a /= (1 - self.dropout_prob)
        else:
            dropout_mask = None
        layer2_z = np.dot(layer1_a, self.layer2_weights) + self.layer2_bias
        layer2_a = self.softmax_function(layer2_z)
        return layer2_a, (layer1_z, layer1_a, dropout_mask, layer2_z, layer2_a)
    
    def backward_pass(self, inputs, true_labels, forward_cache):
        layer1_z, layer1_a, dropout_mask, layer2_z, layer2_a = forward_cache
        num_samples = true_labels.shape[0]
        
        layer2_dz = layer2_a
        layer2_dz[range(num_samples), true_labels] -= 1
        layer2_dz /= num_samples
        
        layer2_dw = np.dot(layer1_a.T, layer2_dz) + (self.reg_rate / num_samples) * self.layer2_weights
        layer2_db = np.sum(layer2_dz, axis=0, keepdims=True)
        
        layer1_da = np.dot(layer2_dz, self.layer2_weights.T)
        if dropout_mask is not None:
            layer1_da *= dropout_mask
            layer1_da /= (1 - self.dropout_prob)
        layer1_dz = self.relu_derivative(layer1_da, layer1_z)
        
        layer1_dw = np.dot(inputs.T, layer1_dz) + (self.reg_rate / num_samples) * self.layer1_weights
        layer1_db = np.sum(layer1_dz, axis=0, keepdims=True)
        
        return layer1_dw, layer1_db, layer2_dw, layer2_db
    
    def update_parameters(self, gradients, learning_rate):
        layer1_dw, layer1_db, layer2_dw, layer2_db = gradients
        self.layer1_weights -= learning_rate * layer1_dw
        self.layer1_bias -= learning_rate * layer1_db
        self.layer2_weights -= learning_rate * layer2_dw
        self.layer2_bias -= learning_rate * layer2_db

def preprocess_data(filepath):
    dataset = pd.read_csv(filepath)
    features = dataset.drop('quality', axis=1).values
    labels = dataset['quality'].values - 3
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    return features_train, features_test, labels_train, labels_test

def standardize_data(train_features, test_features):
    mean_values = np.mean(train_features, axis=0)
    std_values = np.std(train_features, axis=0)
    train_features_standard = (train_features - mean_values) / std_values
    test_features_standard = (test_features - mean_values) / std_values
    return train_features_standard, test_features_standard

def train_wine_quality_classifier(train_features, train_labels, input_size, hidden_size, output_size, learning_rate, regularization, dropout, epochs, batch_size):
    classifier = WineQualityClassifier(input_size, hidden_size, output_size, regularization, dropout)
    for epoch in range(epochs):
        shuffled_indices = np.random.permutation(train_features.shape[0])
        features_shuffled = train_features[shuffled_indices]
        labels_shuffled = train_labels[shuffled_indices]
        for i in range(0, train_features.shape[0], batch_size):
            batch_features = features_shuffled[i:i + batch_size]
            batch_labels = labels_shuffled[i:i + batch_size]
            predicted_labels, cache = classifier.forward_pass(batch_features, training_mode=True)
            loss = classifier.calculate_loss(batch_labels, predicted_labels)
            gradients = classifier.backward_pass(batch_features, batch_labels, cache)
            classifier.update_parameters(gradients, learning_rate)
        print(f"Epoch {epoch + 1}, Loss: {loss}")
    print("Training completed.")

# File path and hyperparameters
file_path = '/content/winequality_red.csv'
features_train, features_test, labels_train, labels_test = preprocess_data(file_path)
features_train, features_test = standardize_data(features_train, features_test)
input_layers = features_train.shape[1]
hidden_layers = 64
output_layers = len(np.unique(labels_train))
learning_rate_value = 0.01
regularization_rate = 0.001
dropout_probability = 0.5
training_epochs = 10
mini_batch_size = 32

# Train the classifier
train_wine_quality_classifier(features_train, labels_train, input_layers, hidden_layers, output_layers, learning_rate_value, regularization_rate, dropout_probability, training_epochs, mini_batch_size)

