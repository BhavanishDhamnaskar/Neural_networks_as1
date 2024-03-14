import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class WineQualityClassifier:
    def __init__(self, input_layers, hidden_layers, output_layers, reg_rate=0.01, drop_prob=0.5):
        self.W1, self.b1 = np.random.randn(input_layers, hidden_layers) * 0.01, np.zeros((1, hidden_layers))
        self.W2, self.b2 = np.random.randn(hidden_layers, output_layers) * 0.01, np.zeros((1, output_layers))
        self.reg_rate, self.drop_prob = reg_rate, drop_prob
        
    def relu(self, Z): return np.maximum(0, Z)
    def d_relu(self, dA, Z): return np.where(Z > 0, dA, 0)
    def softmax(self, Z): expZ = np.exp(Z - np.max(Z)); return expZ / expZ.sum(axis=1, keepdims=True)
    def loss(self, Y, Y_hat): m = Y.size; return -np.log(Y_hat[range(m), Y]).mean() + (self.reg_rate / (2 * m)) * (np.sum(self.W1**2) + np.sum(self.W2**2))
    
    def forward(self, X, train=True):
        Z1, A1 = np.dot(X, self.W1) + self.b1, self.relu(np.dot(X, self.W1) + self.b1)
        A1 *= np.random.binomial(1, 1 - self.drop_prob, A1.shape) / (1 - self.drop_prob) if train else A1
        return self.softmax(np.dot(A1, self.W2) + self.b2), (Z1, A1)
    
    def backward(self, X, Y, cache):
        Z1, A1 = cache; m = Y.size; Y_hat = self.forward(X, False)[0]
        dZ2 = Y_hat; dZ2[range(m), Y] -= 1; dZ2 /= m
        dW2 = np.dot(A1.T, dZ2) + (self.reg_rate / m) * self.W2; db2 = dZ2.mean(axis=0)
        dA1 = np.dot(dZ2, self.W2.T); dZ1 = self.d_relu(dA1, Z1)
        dW1 = np.dot(X.T, dZ1) + (self.reg_rate / m) * self.W1; db1 = dZ1.mean(axis=0)
        return dW1, db1, dW2, db2
    
    def update(self, grads, lr):
        dW1, db1, dW2, db2 = grads
        self.W1 -= lr * dW1; self.b1 -= lr * db1
        self.W2 -= lr * dW2; self.b2 -= lr * db2

def preprocess(filepath):
    df = pd.read_csv(filepath)
    X, y = df.drop('quality', axis=1).values, df['quality'].values - 3
    return train_test_split(X, y, test_size=0.2, random_state=42)

def standardize(train, test): mean, std = train.mean(axis=0), train.std(axis=0); return (train - mean) / std, (test - mean) / std

def train(X, y, input_size, hidden_size, output_size, lr, reg, drop, epochs, batch_size):
    model = WineQualityClassifier(input_size, hidden_size, output_size, reg, drop)
    for epoch in range(epochs):
        for i in range(0, len(y), batch_size):
            X_batch, y_batch = X[i:i + batch_size], y[i:i + batch_size]
            Y_hat, cache = model.forward(X_batch)
            loss = model.loss(y_batch, Y_hat)
            grads = model.backward(X_batch, y_batch, cache)
            model.update(grads, lr)
        print(f"Epoch {epoch + 1}, Loss: {loss}")
    print("Training completed.")

file_path = '/content/winequality_red.csv'
X_train, X_test, y_train, y_test = preprocess(file_path)
X_train, X_test = standardize(X_train, X_test)
train(X_train, y_train, X_train.shape[1], 64, len(np.unique(y_train)), 0.01, 0.001, 0.5, 10, 32)
