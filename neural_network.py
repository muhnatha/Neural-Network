import numpy as np

# activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

# loss functions
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true, y_pred):
    return (y_pred - y_true) / y_true.shape[0]

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-9
    return -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))

def binary_cross_entropy_derivative(y_true, y_pred):
    epsilon = 1e-9
    return (y_pred - y_true) / ((y_pred + epsilon) * (1 - y_pred + epsilon))

class NeuralNetwork:
    def __init__(self, layers, activation="sigmoid", loss="mse", lr=0.01):
        self.layers = layers
        self.lr = lr
        self.weights = []
        self.biases = []

        # Choose activation function
        if activation == "sigmoid":
            self.activation, self.activation_derivative = sigmoid, sigmoid_derivative
        elif activation == "relu":
            self.activation, self.activation_derivative = relu, relu_derivative
        elif activation == "tanh":
            self.activation, self.activation_derivative = tanh, tanh_derivative

        # Choose loss function
        if loss == "mse":
            self.loss, self.loss_derivative = mse, mse_derivative
        elif loss == "bce":
            self.loss, self.loss_derivative = binary_cross_entropy, binary_cross_entropy_derivative

        # Initialize weights and biases (Medium style)
        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i+1], layers[i]) * 0.01)
            self.biases.append(np.zeros((layers[i+1], 1)))

    def forward(self, X):
        """X shape: (n_features, n_samples)"""
        self.a = [X]
        self.z = []

        A = X
        for i in range(len(self.weights)):
            Z = self.weights[i] @ A + self.biases[i]   # W @ A + b
            self.z.append(Z)

            if i < len(self.weights) - 1:
                A = self.activation(Z)
            else:
                A = Z  # linear output for regression
            self.a.append(A)

        return A

    def backward(self, Y):
        grads_w = []
        grads_b = []

        m = Y.shape[1]  # number of samples
        error = self.loss_derivative(Y, self.a[-1]) / m

        for i in reversed(range(len(self.weights))):
            if i < len(self.weights) - 1:
                dz = error * self.activation_derivative(self.z[i])
            else:
                dz = error

            dw = dz @ self.a[i].T
            db = np.sum(dz, axis=1, keepdims=True)
            error = self.weights[i].T @ dz

            grads_w.insert(0, dw)
            grads_b.insert(0, db)

        # Gradient descent update
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * grads_w[i]
            self.biases[i] -= self.lr * grads_b[i]

    def train(self, X, Y, epochs=100, batch_size=None):
        n_samples = X.shape[1]  # samples are columns

        for epoch in range(epochs):
            if batch_size:
                for i in range(0, n_samples, batch_size):
                    X_batch = X[:, i:i+batch_size]
                    Y_batch = Y[:, i:i+batch_size]
                    y_pred = self.forward(X_batch)
                    self.backward(Y_batch)
            else:
                y_pred = self.forward(X)
                self.backward(Y)

            if epoch % 10 == 0:
                y_pred = self.forward(X)
                print(f"Epoch {epoch}, Loss: {self.loss(Y, y_pred)}")

    def predict(self, X):
        return self.forward(X)
            
