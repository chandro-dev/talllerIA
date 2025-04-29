import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x_vals = np.random.uniform(0, 2*np.pi, 10)
y_vals = np.random.uniform(0, 2*np.pi, 10)
z_vals = np.random.uniform(-1, 1, 10)

X = np.column_stack((x_vals, y_vals, z_vals))
y = np.sin(x_vals) + np.cos(y_vals) + z_vals

model = Sequential([
    Dense(10, activation='relu', input_shape=(3,)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=100, verbose=0)

print("Predicción para un nuevo valor:", model.predict(np.array([[1, 1, 0]])))