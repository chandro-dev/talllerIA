import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
import pandas as pd

# Datos binarios (síntomas) y etiquetas (resfriado o no)
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
# Funciones RBF
# -------------------------------------
def rbf_features(X, centers, gamma):
    return rbf_kernel(X, centers, gamma=gamma)

def train_rbf(X, y, n_centers=4, gamma=1.0):
    kmeans = KMeans(n_clusters=n_centers, random_state=0).fit(X)
    centers = kmeans.cluster_centers_
    Phi = rbf_features(X, centers, gamma)
    W = np.linalg.pinv(Phi).dot(y)
    return W, centers

def predict_rbf(X_new, W, centers, gamma):
    Phi = rbf_features(X_new, centers, gamma)
    return Phi.dot(W)

# -------------------------------------
# Entrenar modelo
# -------------------------------------
gamma = 1.0
n_centers = 4
W, centers = train_rbf(X_train, y_train, n_centers, gamma)

# -------------------------------------
# Diagnóstico para una entrada
# -------------------------------------
sintomas = ["Dolor de cabeza", "Fiebre", "Tos", "Dolor de rodilla"]
entrada_ejemplo = [1, 0, 0, 0]

def diagnosticar_resfriado_rbf(entrada_binaria):
    entrada = np.array([entrada_binaria])
    pred = predict_rbf(entrada, W, centers, gamma)
    return 1 if pred[0] > 0.5 else 0, pred[0]

# Diagnóstico de ejemplo
diagnostico, valor_predicho = diagnosticar_resfriado_rbf(entrada_ejemplo)

print("Síntomas ingresados:")
for i, val in enumerate(entrada_ejemplo):
    print(f"  {sintomas[i]}: {'Sí' if val == 1 else 'No'}")

print("\nDiagnóstico:")
print(f"  {'Resfriado' if diagnostico == 1 else 'Sin resfriado'} (valor de salida: {valor_predicho:.4f})")

# -------------------------------------
# Ver todas las predicciones
# -------------------------------------
valores_pred = predict_rbf(X_train, W, centers, gamma)
tabla = pd.DataFrame({
    "Síntomas": list(map(tuple, X_train)),
    "Real": y_train,
    "Predicho": (valores_pred > 0.5).astype(int),
    "Valor RBF": np.round(valores_pred, 4)
})

print("\nResultados sobre todos los datos:")
print(tabla)
