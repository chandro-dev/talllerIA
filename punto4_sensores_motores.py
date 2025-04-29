import numpy as np
from sklearn.linear_model import Perceptron, SGDRegressor

X = np.array([
    [1, 1, 1],
    [-1, 1, 1],
    [1, 1, -1],
    [-1, -1, -1]
])
y = np.array([
    [-1, -1],
    [-1, 1],
    [1, -1],
    [1, 1]
])

p1 = Perceptron()
p1.fit(X, y[:, 0])

a2 = SGDRegressor(learning_rate='constant', eta0=0.01)
a2.fit(X, y[:, 1])

print("Predicción M1:", p1.predict([[1, -1, 1]]))
print("Predicción M2:", a2.predict([[1, -1, 1]]))