import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# mnist Datnsatz importieren und einmal testweise printen (60.000 Bilder jeweils 28x28 Pixel)
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)  # Bilder
print(y_train.shape)  # Labels 

# TTrainingsbilder und Testbilder anzeigen (jeweils die ersten 5)
def visualize_samples(X, y, n = 5):
    plt.figure(figsize=(10,2))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(X[i], cmap = 'gray')
        plt.title(f"Label: {y[i]}")
        plt.axis('off')
    plt.show()

print("Hier ein paar Beispielbilder aus dem MNIST-Datensatz:")
visualize_samples(X_train, y_train)

# Bilder auf Werte zwischen 0 und 1 normalisieren (macht das Training stabiler)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Bilder in flache Vektoren umwandeln (aus 28x28 wird 784)
X_train_flat = X_train.reshape(X_train.shape[0], -1).T  # Shape: (784, 60000)
X_test_flat = X_test.reshape(X_test.shape[0], -1).T      # Shape: (784, 10000)

# ReLU-Aktivierungsfunktion returned Z wenn > 0 sonst returned 0 
def ReLU(Z):
    return np.maximum(Z,0)

# Ableitung von ReLU returned 1 für Z > 0 sonst 0
def ReLU_deriv(Z):
    return Z > 0  # True wenn Z > 0, sonst False (0)

# Softmax-Aktivierungsfunktion (macht Wahrscheinlichkeiten draus)
def softmax(Z):
    return np.exp(Z) / sum(np.exp(Z))

# Initialisiert die Gewichte und Biases für die beiden Hidden Layers
def init_params():
    # Gewichte zufällig zwischen -0.5 und 0.5 setzen
    W1 = np.random.rand(10, 784) - 0.5  # H1: 10 Neuronen, 784 Inputs
    b1 = np.random.rand(10, 1) - 0.5  
    W2 = np.random.rand(10, 10) - 0.5  # H2: 10 Neuronen, 10 Inputs von H1
    b2 = np.random.rand(10, 1) - 0.5 
    return W1, b1, W2, b2

# Vorwärtsdurchlauf H1 -> A1 mit ReLU, H2 -> A2 mit Softmax
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)  # Output von H1 nach Aktivierung
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)  # Output von H2 nach Aktivierung
    return Z1, A1, Z2, A2

# Encoding für die Labels (z.B. 3 = [0,0,0,1,0,0,0,0,0,0])
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))  # Matrix mit Nullen (Anzahl Labels x 10 Klassen)
    one_hot_Y[np.arange(Y.size), Y] = 1  # Setzt für jedes Label die passende 1
    one_hot_Y = one_hot_Y.T  # Transponieren, damit es passt
    return one_hot_Y

# Rückwärtsdurchlauf (Backpropagation)
def back_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    # Der Loss ist hier schon im Softmax drin, daher kein extra Loss nötig
    m = Y.size  # Anzahl der Datenpunkte (meist 60.000)
    one_hot_Y = one_hot(Y) 
    dZ2 = A2 - one_hot_Y  # Differenz zwischen Vorhersage und echtem Wert
    dW2 = 1 / m * dZ2.dot(A1.T)  # Gradienten für W2
    db2 = 1 / m * np.sum(dZ2)  # Gradienten für b2
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

# Gewichte und Biases updaten
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

# Gibt die Klasse mit der höchsten Wahrscheinlichkeit zurück
def get_predictions(A2):
    return np.argmax(A2, 0)

# Einfache Accuracy-Funktion
def get_accuracy(predictions, Y):
    print("Vorhersagen:", predictions)
    print("Echte Labels:", Y)
    return np.sum(predictions == Y) / Y.size

# Klassischer Gradientenabstieg (Batch Gradient Descent)
def gradient_descent(X ,Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        
        if (i % 10 == 0):  # Alle 10 Durchläufe mal ausgeben
            print(f"Iteration {i}:")
            print("Aktuelle Accuracy:", get_accuracy(get_predictions(A2), Y))
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(X_train_flat, y_train, 500, 0.1)

'''Eignetlich unnötig aber kleine visualisierung, um zu sehen wie das Modell so performt (vor allem am Anfang)
oder für präsentieren'''
def visualize_random_predictions(X_test, y_test, model, num_samples=10):
    # Zufällige Indizes auswählen
    random_indices = np.random.choice(X_test.shape[1], num_samples, replace=False)

    # Bilder und Labels rauspicken
    X_random = X_test[:, random_indices]
    y_random = y_test[random_indices]

    # Vorwärtsdurchlauf
    Z1, A1, Z2, A2 = forward_prop(model['W1'], model['b1'], model['W2'], model['b2'], X_random)
    predictions = get_predictions(A2)

    # Bilder anzeigen mit Vorhersage und echtem Wert
    plt.figure(figsize=(10, 2))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(X_random[:, i].reshape(28, 28), cmap='gray')
        pred_label = predictions[i]
        true_label = y_random[i]
        plt.title(f"Vorh: {pred_label}\nEcht: {true_label}")
        plt.axis('off')
    plt.show()

visualize_random_predictions(X_test_flat, y_test, {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2})

# One-hot-Encoding für SGD (einzelnes Label)
def one_hot_sgd(Y):
    one_hot_Y = np.zeros((10,1))  # 10 Klassen
    one_hot_Y[Y] = 1
    return one_hot_Y

''''Wie im Tagebuch erwähnt itterieren wir hier einfach durch die sample um dem modell zu sagen was es ändern soll 
(also in Fachtermen Backpropagation)'''
def back_prop_sgd(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = 1  # Wir machen SGD, also immer nur ein Sample
    one_hot_Y = one_hot_sgd(Y)  # Label one-hot encoden
    dZ2 = A2 - one_hot_Y  # Differenz zwischen Vorhersage und echtem Wert
    dW2 = 1 / m * dZ2.dot(A1.T)  # Gradienten für W2
    db2 = 1 / m * np.sum(dZ2)  # Gradienten für b2
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)  # Backprop zur ersten Schicht
    dW1 = 1 / m * dZ1.dot(X.T)  # Gradienten für W1
    db1 = 1 / m * np.sum(dZ1)  # Gradienten für b1
    return dW1, db1, dW2, db2

# Stochastic Gradient Descent (SGD)
def sgd(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()  
    for i in range(iterations):  
        for j in range(X.shape[1]):  # Über jedes Sample einzeln gehen 
            X_single = X[:, j:j+1]  # Ein einzelnes Bild
            Y_single = Y[j]  # Das zugehörige Label
            # Vorwärtsdurchlauf
            Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_single)
            # Backpropagation
            dW1, db1, dW2, db2 = back_prop_sgd(Z1, A1, Z2, A2, W1, W2, X_single, Y_single)
            # Parameter updaten
            W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        # Alle 10 Durchläufe Accuracy checken
        if (i % 10 == 0):
            print(f"Iteration {i}:")
            print("Aktuelle Accuracy:", get_accuracy(get_predictions(A2), Y))
    # Am Ende die Parameter zurückgeben
    return W1, b1, W2, b2

W1, b1, W2, b2 = sgd(X_train_flat, y_train, 500, 0.1)
