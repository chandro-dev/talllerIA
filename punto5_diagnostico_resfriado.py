import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel

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
# RBF Helpers
# -------------------------------------
def rbf_features(X, centers, gamma):
    return rbf_kernel(X, centers, gamma=gamma)

def train_rbf(X, y, n_centers=3, gamma=1.0):
    kmeans = KMeans(n_clusters=n_centers, random_state=0).fit(X)
    centers = kmeans.cluster_centers_
    Phi = rbf_features(X, centers, gamma)
    W = np.linalg.pinv(Phi).dot(y)
    return W, centers

def predict_rbf(X_new, W, centers, gamma):
    Phi = rbf_features(X_new, centers, gamma)
    return Phi.dot(W)

# -------------------------------------
# Entrenamiento del modelo RBF
# -------------------------------------
gamma = 1.0
n_centers = 3
W, centers = train_rbf(X_train, y_train, n_centers, gamma)

# -------------------------------------
# Diagnóstico con una entrada de ejemplo
# -------------------------------------
sintomas = ["Dolor de cabeza", "Fiebre", "Tos", "Dolor de rodilla"]
entrada_ejemplo = [1, 0,0,0]  # Síntomas

def diagnosticar_resfriado_rbf(entrada_binaria):
    entrada = np.array([entrada_binaria])
    pred = predict_rbf(entrada, W, centers, gamma)
    return 1 if pred[0] > 0.5 else 0, pred[0]

diagnostico, valor_predicho = diagnosticar_resfriado_rbf(entrada_ejemplo)

# Mostrar resultados
print("Síntomas ingresados:")
for i, val in enumerate(entrada_ejemplo):
    print(f"  {sintomas[i]}: {'Sí' if val == 1 else 'No'}")

print("\nDiagnóstico:")
print(f"  {'Resfriado' if diagnostico == 1 else 'Sin resfriado'} (valor de salida: {valor_predicho:.4f})")