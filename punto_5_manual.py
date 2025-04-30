import numpy as np

# -------------------------------------
# Datos de entrenamiento (entradas y salidas)
# -------------------------------------
X_train = np.array([
    [0, 0, 0, 0],
    [1, 1, 1, 1],
    [1, 1, 1, 0],
    [0, 0, 0, 1],
    [0, 1, 1, 0],
    [0, 1, 1, 1],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [1, 0, 1, 0],
    [1, 0, 1, 1]
])
y_train = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 1])


# -------------------------------------
# Definir el número de centros radiales 4
# Inicializar los centros aleatoriamente
# -------------------------------------
cr = np.array([
    [1.3, 0.3, 0.8, 1.4],
    [1.1, 0.2, 0.4, 1.2],
    [0.9, 0.5, 1.0, 0.7],
    [1.5, 0.3, 0.6, 1.3]
])

# -------------------------------------
# Calcular la distancia euclidiana entre los puntos de entrada y los centros
# Di = (sum((x-cr)^2))^(1/2)
# -------------------------------------
distancias = np.sqrt(np.sum((X_train[:, np.newaxis] - cr) ** 2, axis=2))

# -------------------------------------
# Calcular la función de activación RBF
# FA = Di^2 * ln(Di)
# -------------------------------------
funciones_activacion = np.exp(-distancias ** 2 / (2 * 0.1 ** 2))


# -------------------------------------



