import numpy as np

class ActivationFunction:
    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=0))
        return e_x / e_x.sum(axis=0)

    @staticmethod
    def d_relu(x):
        return np.where(x > 0, 1, 0)

    @staticmethod
    def d_sigmoid(x):
        s = 1 / (1 + np.exp(-x))
        return s * (1 - s)

    @staticmethod
    def d_tanh(x):
        return 1 - np.tanh(x) ** 2


class Layer:
    def __init__(self, input_dim, output_dim, activation, learning_rate=0.01):
        self.weights = np.random.randn(output_dim, input_dim) * 0.01
        self.bias = np.zeros((output_dim, 1))
        self.activation = activation
        self.learning_rate = learning_rate
        self.Z = None
        self.A = None

    def forward(self, A_prev):
        self.Z = np.dot(self.weights, A_prev) + self.bias
        self.A = getattr(ActivationFunction, self.activation)(self.Z)
        return self.A

    def backward(self, dA, A_prev):
        m = A_prev.shape[1]
        if self.activation == 'softmax':
            dZ = dA
        else:
            dZ = dA * ActivationFunction.d_relu(self.Z)  # Use the appropriate derivative function

        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(self.weights.T, dZ)

        self.weights -= self.learning_rate * dW
        self.bias -= self.learning_rate * db

        return dA_prev

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, input_dim, output_dim, activation):
        layer = Layer(input_dim, output_dim, activation)
        self.layers.append(layer)

    def forward_propagation(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def compute_loss(self, Y_pred, Y_true):
        m = Y_true.shape[1]
        loss = -np.sum(Y_true * np.log(Y_pred)) / m
        return loss

    def backpropagation(self, Y_pred, Y_true, X):
        m = Y_true.shape[1]
        dA = Y_pred - Y_true  # For softmax with cross-entropy loss

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            A_prev = X if i == 0 else self.layers[i - 1].A
            dA = layer.backward(dA, A_prev)

    def train(self, X, Y, epochs):
        for epoch in range(epochs):
            Y_pred = self.forward_propagation(X)
            loss = self.compute_loss(Y_pred, Y)
            self.backpropagation(Y_pred, Y, X)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

# Example usage
if __name__ == '__main__':
    nn = NeuralNetwork()
    nn.add_layer(input_dim=2, output_dim=4, activation='relu')
    nn.add_layer(input_dim=4, output_dim=2, activation='softmax')

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input
    Y = np.array([[1, 0], [0, 1], [0, 1], [1,1]])
    
# Output

nn.train(X.T,Y.T, epochs=1000)
