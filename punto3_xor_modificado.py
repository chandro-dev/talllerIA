import numpy as np
from sklearn.neural_network import MLPClassifier

X = np.array([
    [-1, -1],
    [-1, 1],
    [1, -1],
    [1, 1]
])
y = np.array([-1, -1, -1, 1])

clf = MLPClassifier(hidden_layer_sizes=(4,), activation='tanh', max_iter=1000)
clf.fit(X, y)

print("Predicción XOR modificada:", clf.predict([[1, 1]]))