import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================
# MNIST lokal laden (Binary Files)
# =========================

DATA_PATH = "/Users/Guest/PyCharmMiscProject/data"

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

# Daten laden
X_train = load_mnist_images(os.path.join(DATA_PATH, "train-images.idx3-ubyte"))
y_train = load_mnist_labels(os.path.join(DATA_PATH, "train-labels.idx1-ubyte"))

X_test = load_mnist_images(os.path.join(DATA_PATH, "t10k-images.idx3-ubyte"))
y_test = load_mnist_labels(os.path.join(DATA_PATH, "t10k-labels.idx1-ubyte"))

print(X_train.shape)
print(y_train.shape)


# =========================
# Visualisierung
# =========================

def visualize_samples(X, y, n=5):
    plt.figure(figsize=(10, 2))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(X[i], cmap='gray')
        plt.title(f"Label: {y[i]}")
        plt.axis('off')
    plt.show()

print("Hier ein paar Beispielbilder aus dem MNIST-Datensatz:")
visualize_samples(X_train, y_train)


# =========================
# Preprocessing
# =========================

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train_flat = X_train.reshape(X_train.shape[0], -1).T
X_test_flat = X_test.reshape(X_test.shape[0], -1).T


# =========================
# Aktivierungsfunktionen
# =========================

def ReLU(Z):
    return np.maximum(Z, 0)

def ReLU_deriv(Z):
    return Z > 0

def softmax(Z):
    Z = Z - np.max(Z, axis=0, keepdims=True)  # stabil
    return np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)


# =========================
# Initialisierung
# =========================

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2


# =========================
# Forward Propagation
# =========================

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


# =========================
# One-Hot Encoding
# =========================

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T


# =========================
# Backpropagation (Batch)
# =========================

def back_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)

    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)

    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)

    return dW1, db1, dW2, db2


# =========================
# Parameter Update
# =========================

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    return W1, b1, W2, b2


# =========================
# Accuracy
# =========================

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print("Vorhersagen:", predictions)
    print("Echte Labels:", Y)
    return np.sum(predictions == Y) / Y.size


# =========================
# Gradient Descent
# =========================

def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        if i % 1 == 0:
            print(f"Iteration {i}:")
            print("Accuracy:", get_accuracy(get_predictions(A2), Y))

    return W1, b1, W2, b2


W1, b1, W2, b2 = gradient_descent(X_train_flat, y_train, 500, 0.35)


# =========================
# Visualisierung Predictions
# =========================

def visualize_random_predictions(X_test, y_test, model, num_samples=10):
    random_indices = np.random.choice(X_test.shape[1], num_samples, replace=False)

    X_random = X_test[:, random_indices]
    y_random = y_test[random_indices]

    Z1, A1, Z2, A2 = forward_prop(model['W1'], model['b1'], model['W2'], model['b2'], X_random)
    predictions = get_predictions(A2)

    plt.figure(figsize=(10, 2))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(X_random[:, i].reshape(28, 28), cmap='gray')
        plt.title(f"Pred: {predictions[i]}\nReal: {y_random[i]}")
        plt.axis('off')
    plt.show()


visualize_random_predictions(
    X_test_flat,
    y_test,
    {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
)


# =========================
# SGD
# =========================

def one_hot_sgd(Y):
    one_hot_Y = np.zeros((10, 1))
    one_hot_Y[Y] = 1
    return one_hot_Y


def back_prop_sgd(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot_sgd(Y)

    dZ2 = A2 - one_hot_Y
    dW2 = dZ2.dot(A1.T)
    db2 = np.sum(dZ2)

    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = dZ1.dot(X.T)
    db1 = np.sum(dZ1)

    return dW1, db1, dW2, db2


def sgd(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()

    for i in range(iterations):
        for j in range(X.shape[1]):
            X_single = X[:, j:j + 1]
            Y_single = Y[j]

            Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_single)
            dW1, db1, dW2, db2 = back_prop_sgd(Z1, A1, Z2, A2, W1, W2, X_single, Y_single)

            W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        if i % 10 == 0:
            print(f"SGD Iteration {i}")

    return W1, b1, W2, b2


W1, b1, W2, b2 = sgd(X_train_flat, y_train, 500, 0.2)
