
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

CSV_MAIN   = "kidney_disease_dataset.csv"
CSV_TEST   = "erc_test_data.csv"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS     = 100
LR         = 1e-3

# ----------------------------
# Load & preprocess data
# ----------------------------
df = pd.read_csv(CSV_MAIN)
encoders = {}
for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df.drop("Target", axis=1).values.astype(np.float32)
y = df["Target"].values.astype(np.int64)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
val_dataset   = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE)

# ----------------------------
# Define MLP model
# ----------------------------
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.model(x)

input_dim = X.shape[1]
n_classes = len(np.unique(y))
model = MLPClassifier(input_dim, n_classes).to(DEVICE)

# ----------------------------
# Training setup
# ----------------------------
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weight_tensor)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

# ----------------------------
# Training loop
# ----------------------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        output = model(xb)
        loss = criterion(output, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    avg_loss = total_loss / len(train_loader.dataset)

    # Validation
    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
    val_acc = correct / len(val_loader.dataset)

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

# ----------------------------
# Evaluation
# ----------------------------
model.eval()
all_preds, all_true = [], []
with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(DEVICE)
        preds = model(xb).argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_true.append(yb.numpy())

y_pred = np.concatenate(all_preds)
y_true = np.concatenate(all_true)
target_names = encoders["Target"].classes_
print("\nClassification report (validation):")
print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(xticks_rotation=45)
plt.title("Matriz de Confusión (Validación)")
plt.show()

# ----------------------------
# Prediction on external test set
# ----------------------------
df_ext = pd.read_csv(CSV_TEST)
for col, le in encoders.items():
    df_ext[col] = le.transform(df_ext[col])

X_ext = scaler.transform(df_ext.drop("Target", axis=1).values.astype(np.float32))
X_ext_t = torch.tensor(X_ext, dtype=torch.float32, device=DEVICE)

model.eval()
with torch.no_grad():
    pred_ext = model(X_ext_t).argmax(dim=1).cpu().numpy()

labels_ext = encoders["Target"].inverse_transform(pred_ext)
df_ext["Predicción"] = labels_ext
df_ext["Target"] = encoders["Target"].inverse_transform(df_ext["Target"])
print("\nResultados externos Actual vs Predicción:")
print(df_ext[["Target", "Predicción"]])
