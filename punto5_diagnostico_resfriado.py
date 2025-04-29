import numpy as np
from sklearn.linear_model import Perceptron

X = np.array([
    [0,0,0,0],
    [1,1,1,1],
    [1,1,1,0],
    [0,0,0,1],
    [0,1,1,0],
    [0,1,1,1],
    [0,0,1,0],
    [0,0,1,1],
    [1,0,1,0],
    [1,0,1,1]
])
y = np.array([0,1,1,0,1,1,0,0,1,1])

model = Perceptron()
model.fit(X, y)

print("Predicción para [1,1,1,0]:", model.predict([[1,1,1,0]]))