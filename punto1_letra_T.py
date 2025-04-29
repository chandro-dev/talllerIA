import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

letras = {
    'A': [2, 2, 4, 4, 10, 10, 4, 4, 2, 2],
    'B': [10, 10, 6, 6, 10, 10, 6, 6, 10, 10],
    'T': [2, 2, 2, 2, 10, 10, 2, 2, 2, 2]
}

X = np.array(list(letras.values()))
y = np.array([0, 1, 2])

y_cat = to_categorical(y, num_classes=3)

model = Sequential([
    Dense(16, activation='relu', input_shape=(10,)),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y_cat, epochs=100, verbose=0)

pred = model.predict(np.array([[2, 2, 2, 2, 10, 10, 2, 2, 2, 2]]))
print("Letra reconocida (índice):", np.argmax(pred))