import numpy as np
import pandas as pd
from numpy.linalg import pinv

# -------------------------------------
# Datos de entrenamiento (entradas binarias y etiquetas)
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
# Definición manual de centros
# -------------------------------------
centros = np.array([
    [1.3, 0.3, 0.8, 1.4],
    [1.1, 0.2, 0.4, 1.2],
    [0.9, 0.5, 1.0, 0.7],
    [1.5, 0.3, 0.6, 1.3],
        [1.3, 0.3, 0.8, 1.4],
    [1.1, 0.2, 0.4, 1.2],
        [1.3, 0.3, 0.8, 1.4],
    [1.1, 0.2, 0.4, 1.2]
])

# -------------------------------------
# Calcular distancias y activaciones FA = d² · ln(d)
# -------------------------------------
distancias = np.sqrt(np.sum((X_train[:, np.newaxis] - centros) ** 2, axis=2))

with np.errstate(divide='ignore', invalid='ignore'):
    Phi = distancias ** 2 * np.log(distancias)
    Phi[np.isnan(Phi)] = 0.0  # evitar log(0)

# -------------------------------------
# Entrenamiento: resolver pesos W
# -------------------------------------
W = pinv(Phi).dot(y_train)

# -------------------------------------
# Diagnóstico para una entrada personalizada
# -------------------------------------
entrada_ejemplo = np.array([[1, 0, 0, 0]])
dist_ejemplo = np.sqrt(np.sum((entrada_ejemplo - centros) ** 2, axis=1))

with np.errstate(divide='ignore', invalid='ignore'):
    phi_ejemplo = dist_ejemplo ** 2 * np.log(dist_ejemplo)
    phi_ejemplo[np.isnan(phi_ejemplo)] = 0.0

valor_predicho = phi_ejemplo.dot(W)
diagnostico = 1 if valor_predicho > 0.5 else 0

sintomas = ["Dolor de cabeza", "Fiebre", "Tos", "Dolor de rodilla"]
print("Síntomas ingresados:")
for i, val in enumerate(entrada_ejemplo[0]):
    print(f"  {sintomas[i]}: {'Sí' if val == 1 else 'No'}")

print("\nDiagnóstico:")
print(f"  {'Resfriado' if diagnostico == 1 else 'Sin resfriado'} (valor de salida: {valor_predicho:.4f})")

# -------------------------------------
# Evaluación sobre todos los datos
# -------------------------------------
Phi_all = distancias ** 2 * np.log(distancias)
Phi_all[np.isnan(Phi_all)] = 0.0
pred_all = Phi_all.dot(W)

tabla = pd.DataFrame({
    "Síntomas": list(map(tuple, X_train)),
    "Real": y_train,
    "Predicho": (pred_all > 0.5).astype(int),
    "Valor RBF log": np.round(pred_all, 4)
})

print("\nResultados sobre el dataset completo:")
print(tabla)
