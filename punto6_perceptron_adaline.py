import numpy as np
from sklearn.linear_model import SGDRegressor

X = np.array([
    [0.1, 0.9, 0.1],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.1],
    [0.9, 0.9, 0.9]
])
y1 = np.array([0.1, 0.1, 0.1, 0.9])
y2 = np.array([0.9, 0.9, 0.1, 0.9])

adaline1 = SGDRegressor(eta0=0.1, learning_rate='constant')
adaline2 = SGDRegressor(eta0=0.1, learning_rate='constant')

adaline1.fit(X, y1)
adaline2.fit(X, y2)

print("Predicción d1:", adaline1.predict([[0.1, 0.9, 0.1]]))
print("Predicción d2:", adaline2.predict([[0.1, 0.9, 0.1]]))