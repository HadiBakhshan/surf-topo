# # forward_model_train.py
# import h5py
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset

# # ==============================
# # Load dataset
# # ==============================
# file_path = "D:/Codes/surf-topo/dataset_simulation/dataset.h5"  # replace with your dataset path

# with h5py.File(file_path, "r") as f:
#     X = f["X"][:].astype(np.float32)
#     Y = f["Y"][:].astype(np.float32)

# # Convert to PyTorch tensors
# X_tensor = torch.from_numpy(X)
# Y_tensor = torch.from_numpy(Y)

# dataset = TensorDataset(X_tensor, Y_tensor)
# dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# # ==============================
# # Define Forward DNN Model
# # ==============================
# class ForwardDNN(nn.Module):
#     def __init__(self, input_dim=6, output_dim=3):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Linear(input_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, output_dim)
#         )
#     def forward(self, x):
#         return self.model(x)

# # ==============================
# # Training
# # ==============================
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = ForwardDNN().to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# criterion = nn.MSELoss()

# n_epochs = 50

# for epoch in range(n_epochs):
#     epoch_loss = 0
#     for xb, yb in dataloader:
#         xb, yb = xb.to(device), yb.to(device)
#         optimizer.zero_grad()
#         y_pred = model(xb)
#         loss = criterion(y_pred, yb)
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item() * xb.size(0)
#     epoch_loss /= len(dataset)
#     if (epoch+1) % 20 == 0:
#         print(f"Epoch {epoch+1}/{n_epochs} | Loss: {epoch_loss:.6f}")

# # ==============================
# # Save trained model
# # ==============================
# torch.save(model.state_dict(), "forward_dnn.pth")
# print("Forward model saved as forward_dnn.pth")

# forward_model_train.py


import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ==============================
# Load dataset
# ==============================

file_path = "D:/Codes/surf-topo/dataset_simulation/dataset.h5"

with h5py.File(file_path, "r") as f:
    X = f["X"][:].astype(np.float32)
    Y = f["Y"][:].astype(np.float32)

print("Dataset loaded")
print("X shape:", X.shape)
print("Y shape:", Y.shape)

# ==============================
# Normalize inputs
# ==============================

X_mean = X.mean(axis=0)
X_std = X.std(axis=0) + 1e-8

X = (X - X_mean) / X_std

np.save("X_mean.npy", X_mean)
np.save("X_std.npy", X_std)

print("Input normalization parameters saved")

# ==============================
# Normalize outputs
# ==============================

Y_mean = Y.mean(axis=0)
Y_std = Y.std(axis=0) + 1e-8

Y = (Y - Y_mean) / Y_std

np.save("Y_mean.npy", Y_mean)
np.save("Y_std.npy", Y_std)

print("Output normalization parameters saved")

# ==============================
# Convert to PyTorch tensors
# ==============================

X_tensor = torch.from_numpy(X)
Y_tensor = torch.from_numpy(Y)

dataset = TensorDataset(X_tensor, Y_tensor)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

# ==============================
# Define Forward Model
# ==============================

class ForwardDNN(nn.Module):

    def __init__(self, input_dim, output_dim):

        super().__init__()

        self.model = nn.Sequential(

            nn.Linear(input_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, output_dim)

        )

    def forward(self, x):
        return self.model(x)


input_dim = X.shape[1]
output_dim = Y.shape[1]

print("Input dimension:", input_dim)
print("Output dimension:", output_dim)

# ==============================
# Training setup
# ==============================

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ForwardDNN(input_dim, output_dim).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-5
)

criterion = nn.MSELoss()

n_epochs = 200

# ==============================
# Training loop
# ==============================

print("\nStarting training...\n")

for epoch in range(n_epochs):

    epoch_loss = 0

    for xb, yb in dataloader:

        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        y_pred = model(xb)

        loss = criterion(y_pred, yb)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item() * xb.size(0)

    epoch_loss /= len(dataset)

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{n_epochs} | Loss: {epoch_loss:.6f}")

# ==============================
# Save trained model
# ==============================

torch.save(model.state_dict(), "forward_dnn.pth")

print("\nTraining complete")
print("Model saved as: forward_dnn.pth")

print("\nSaved normalization files:")
print("X_mean.npy")
print("X_std.npy")
print("Y_mean.npy")
print("Y_std.npy")