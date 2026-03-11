import torch
import torch.nn as nn
import numpy as np

# ==============================
# Forward model definition
# ==============================

class ForwardDNN(nn.Module):

    def __init__(self, input_dim=6, output_dim=3):

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


# ==============================
# Load model and normalization
# ==============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = r"D:/Codes/surf-topo/training/forward_dnn.pth"

X_mean_path = r"D:/Codes/surf-topo/training/X_mean.npy"
X_std_path  = r"D:/Codes/surf-topo/training/X_std.npy"

Y_mean_path = r"D:/Codes/surf-topo/training/Y_mean.npy"
Y_std_path  = r"D:/Codes/surf-topo/training/Y_std.npy"

# Load normalization parameters
X_mean = torch.tensor(np.load(X_mean_path), dtype=torch.float32).to(device)
X_std  = torch.tensor(np.load(X_std_path), dtype=torch.float32).to(device)

Y_mean = torch.tensor(np.load(Y_mean_path), dtype=torch.float32).to(device)
Y_std  = torch.tensor(np.load(Y_std_path), dtype=torch.float32).to(device)

# Load model
model = ForwardDNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print("Model loaded successfully")


# ==============================
# Input machining parameters
# ==============================

vc = 200.0
fz = 0.5
eps_r = -0.026
eps_a = 0.009
z = 4
ri = 3


# ==============================
# Prepare input
# ==============================

x_raw = torch.tensor([[vc, fz, eps_r, eps_a, z, ri]], dtype=torch.float32).to(device)

# Normalize inputs
x = (x_raw - X_mean) / X_std


# ==============================
# Run prediction
# ==============================

with torch.no_grad():

    pred_norm = model(x)

    # Denormalize outputs
    pred = pred_norm * Y_std + Y_mean

    pred = pred.cpu().numpy()[0]


# ==============================
# Print results
# ==============================

print("\n==============================")
print("INPUT PARAMETERS")
print("==============================")

print(f"vc   = {vc}")
print(f"fz   = {fz}")
print(f"eps_r= {eps_r}")
print(f"eps_a= {eps_a}")
print(f"z    = {z}")
print(f"ri   = {ri}")

print("\n==============================")
print("PREDICTED SURFACE METRICS")
print("==============================")

print(f"Ra = {pred[0]:.4f}")
print(f"Rq = {pred[1]:.4f}")
print(f"Rz = {pred[2]:.4f}")