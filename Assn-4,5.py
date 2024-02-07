1. Activation Class
import numpy as np

class Activation:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return Activation.sigmoid(x) * (1 - Activation.sigmoid(x))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1.0, 0.0)


2. Layer Class
class Layer:
    def __init__(self, input_size, output_size, activation=None):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.bias = np.zeros((1, output_size))
        self.activation = activation
        self.output = None
        self.input = None

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(input_data, self.weights) + self.bias
        if self.activation is not None:
            self.output = self.activation(self.output)
        return self.output


3. Model Class

class Model:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

4. LossFunction Class

class LossFunction:
    @staticmethod
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def mse_derivative(y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size
      
5. BackProp and GradDescent

def train(model, X_train, y_train, loss_function, lr=0.01, epochs=1000):
    for epoch in range(epochs):
        # Forward pass
        output = model.predict(X_train)
        
        # Compute loss
        loss = loss_function(y_train, output)
        
        # Backward pass (for a simple network, we'll manually compute gradients)
        # This is a simplified version and needs to be expanded based on the network's complexity
        
        # Update model parameters (weights and biases)
        # Placeholder for actual backpropagation and gradient descent logic
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")


6. Class – ForwardProp
class ForwardProp:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def activate(self, x):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        self.Z = np.dot(X, self.weights) + self.bias
        self.A = self.activate(self.Z)
        return self.A

7. Class – BackProp

class BackProp:
    def __init__(self):
        self.dweights = None
        self.dbias = None

    def activate_derivative(self, x):
        # Derivative of sigmoid function
        return x * (1 - x)

    def backward(self, X, Y, A):
        m = X.shape[0]
        dZ = A - Y
        self.dweights = np.dot(X.T, dZ) / m
        self.dbias = np.sum(dZ) / m
        return self.dweights, self.dbias
      
8)  Class – GradDescent

class GradDescent:
    def update(self, weights, bias, dweights, dbias, learning_rate):
        weights -= learning_rate * dweights
        bias -= learning_rate * dbias
        return weights, bias

9) Class – Traning

class Training:
    def __init__(self, X, Y, hidden_size, learning_rate=0.01, epochs=1000):
        self.X = X
        self.Y = Y
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.hidden_size = hidden_size
        self.input_size = X.shape[1]
        self.output_size = Y.shape[1]
        
        # Initialize weights and bias
        self.weights = np.random.randn(self.input_size, self.hidden_size)
        self.bias = np.zeros((1, self.hidden_size))
        
        # Instantiate ForwardProp, BackProp, and GradDescent classes
        self.fp = ForwardProp(self.weights, self.bias)
        self.bp = BackProp()
        self.gd = GradDescent()

    def train(self):
        for i in range(self.epochs):
            # Forward propagation
            A = self.fp.forward(self.X)
            
            # Backpropagation
            dweights, dbias = self.bp.backward(self.X, self.Y, A)
            
            # Gradient descent parameter update
            self.weights, self.bias = self.gd.update(self.fp.weights, self.fp.bias, dweights, dbias, self.learning_rate)
            
            # Optionally, print loss every 100 steps
            if i % 100 == 0:
                loss = -np.mean(self.Y * np.log(A) + (1 - self.Y) * np.log(1 - A))
                print(f"Epoch {i}, Loss: {loss}")


