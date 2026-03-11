# # inverse_design.py
# import torch
# import torch.nn as nn
# import numpy as np

# # ==============================
# # Load trained forward model
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

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = ForwardDNN().to(device)

# # with the full path to the trained model
# model_path = r"D:/Codes/surf-topo/training/forward_dnn.pth"
# model.load_state_dict(torch.load(model_path, map_location=device))
# model.eval()

# # ==============================
# # Target surface quality
# # ==============================
# target = torch.tensor([[0.8, 1.0, 6.0]], dtype=torch.float32).to(device)  # Ra, Rq, Rz

# # ==============================
# # Optimization-based inverse design
# # ==============================
# # Start from random feasible inputs
# # You may set realistic bounds from your dataset
# x_opt = torch.tensor([[200.0, 0.3, -0.02, 0.005, 3, 4]], requires_grad=True, device=device)

# optimizer = torch.optim.Adam([x_opt], lr=1.0)

# # Define parameter bounds
# bounds = torch.tensor([
#     [100, 300],    # vc
#     [0.1, 0.9],    # fz
#     [-0.026, 0.011], # eps_r
#     [0.003, 0.009],  # eps_a
#     [2, 5],        # z
#     [3, 5]         # ri
# ], device=device)

# n_iter = 500

# for i in range(n_iter):
#     optimizer.zero_grad()
#     y_pred = model(x_opt)
#     loss = nn.MSELoss()(y_pred, target)
#     loss.backward()
#     optimizer.step()
    
#     # Clamp parameters to bounds
#     with torch.no_grad():
#         for j in range(x_opt.shape[1]):
#             x_opt[0, j].clamp_(bounds[j,0], bounds[j,1])

#     if (i+1) % 50 == 0:
#         print(f"Iteration {i+1}/{n_iter} | Loss: {loss.item():.6f} | Predicted: {y_pred.detach().cpu().numpy()}")

# print("\n✅ Optimized input parameters:")
# print(x_opt.detach().cpu().numpy())
# print("Predicted surface metrics:")
# print(model(x_opt).detach().cpu().numpy())






# import torch
# import numpy as np
# from skopt import gp_minimize
# from skopt.space import Real, Integer, Categorical

# # -----------------------------
# # Load trained forward model
# # -----------------------------
# class ForwardDNN(torch.nn.Module):
#     def __init__(self, input_dim=6, output_dim=3):
#         super().__init__()
#         self.model = torch.nn.Sequential(
#             torch.nn.Linear(input_dim, 64),
#             torch.nn.ReLU(),
#             torch.nn.Linear(64, 128),
#             torch.nn.ReLU(),
#             torch.nn.Linear(128, 64),
#             torch.nn.ReLU(),
#             torch.nn.Linear(64, output_dim)
#         )

#     def forward(self, x):
#         return self.model(x)

# # Device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load model
# model_path = r"D:/Codes/surf-topo/training/forward_dnn.pth"  # replace with your path
# model = ForwardDNN(input_dim=6, output_dim=3).to(device)
# model.load_state_dict(torch.load(model_path, map_location=device))
# model.eval()

# # -----------------------------
# # Target surface metrics
# # -----------------------------
# target = np.array([1.4, 2.0, 5.6], dtype=np.float32)  # Ra, Rq, Rz

# # -----------------------------
# # Define discrete options
# # -----------------------------
# eps_r_list = [-0.026, -0.013, 0.0, 0.011]
# eps_a_list = [0.003, 0.005, 0.007, 0.009]
# z_list = [2, 3, 4]
# ri_list = [3, 4, 5]

# # Continuous variable bounds
# vc_bounds = (100.0, 300.0)
# fz_bounds = (0.1, 0.9)

# # -----------------------------
# # Define search space for BO
# # -----------------------------
# space = [
#     Real(vc_bounds[0], vc_bounds[1], name="vc"),                  # continuous
#     Real(fz_bounds[0], fz_bounds[1], name="fz"),                  # continuous
#     Categorical(eps_r_list, name="eps_r"),                        # categorical
#     Categorical(eps_a_list, name="eps_a"),                        # categorical
#     Integer(min(z_list), max(z_list), name="z"),                  # integer
#     Integer(min(ri_list), max(ri_list), name="ri")                # integer
# ]

# # -----------------------------
# # Objective function for BO
# # -----------------------------
# def objective(x):
#     # x = [vc, fz, eps_r, eps_a, z, ri]
#     x_tensor = torch.tensor([x], dtype=torch.float32, device=device)
#     with torch.no_grad():
#         y_pred = model(x_tensor).cpu().numpy()[0]
#     # loss = mean squared error between predicted and target metrics
#     loss = np.mean((y_pred - target)**2)
#     return loss

# # -----------------------------
# # Run Bayesian Optimization
# # -----------------------------
# result = gp_minimize(
#     func=objective,
#     dimensions=space,
#     n_calls=100,           # number of evaluations
#     n_initial_points=10,   # initial random samples
#     random_state=42,
#     verbose=True
# )

# # -----------------------------
# # Extract best solution
# # -----------------------------
# best_input = result.x
# best_pred = model(torch.tensor([best_input], dtype=torch.float32, device=device)).detach().cpu().numpy()[0]

# print("\n✅ Best optimized input parameters:")
# print(f"vc = {best_input[0]:.3f}, fz = {best_input[1]:.3f}, "
#       f"eps_r = {best_input[2]}, eps_a = {best_input[3]}, "
#       f"z = {best_input[4]}, ri = {best_input[5]}")

# print("\nPredicted surface metrics [Ra, Rq, Rz]:")
# print(f"Ra = {best_pred[0]:.3f}, Rq = {best_pred[1]:.3f}, Rz = {best_pred[2]:.3f}")

# print(f"\nFinal loss: {result.fun:.6f}")



import torch
import torch.nn as nn
import numpy as np
from itertools import product

# ==============================
# Forward model definition
# ==============================

class ForwardDNN(nn.Module):

    def __init__(self, input_dim=6, output_dim=3):

        super().__init__()

        self.model = nn.Sequential(

            nn.Linear(input_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.model(x)


# ==============================
# Load trained model
# ==============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = r"D:/Codes/surf-topo/training/forward_dnn.pth"
mean_path  = r"D:/Codes/surf-topo/training/X_mean.npy"
std_path   = r"D:/Codes/surf-topo/training/X_std.npy"

# Load normalization parameters
X_mean = torch.tensor(np.load(mean_path), dtype=torch.float32).to(device)
X_std  = torch.tensor(np.load(std_path), dtype=torch.float32).to(device)

model = ForwardDNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


# ==============================
# Target surface metrics
# ==============================

target = torch.tensor([[8.88, 16.6, 47.5]], dtype=torch.float32).to(device)
# target = [Ra, Rq, Rz]


# ==============================
# Continuous variable bounds
# ==============================

vc_bounds = (100.0, 300.0)
fz_bounds = (0.1, 0.9)
eps_r_bounds = (-0.026, 0.011)
eps_a_bounds = (0.003, 0.009)


# ==============================
# Discrete variables
# ==============================

z_list = [2, 3, 4]
ri_list = [3, 4, 5]


# ==============================
# Optimization settings
# ==============================

steps = 500
lr = 0.05

best_loss = float("inf")
best_solution = None


# ==============================
# Loop over discrete variables
# ==============================

for z, ri in product(z_list, ri_list):

    print(f"\nOptimizing for z={z}, ri={ri}")

    # initialize continuous variables
    vc = torch.tensor([200.0], requires_grad=True, device=device)
    fz = torch.tensor([0.4], requires_grad=True, device=device)
    eps_r = torch.tensor([-0.01], requires_grad=True, device=device)
    eps_a = torch.tensor([0.005], requires_grad=True, device=device)

    optimizer = torch.optim.Adam([vc, fz, eps_r, eps_a], lr=lr)

    for step in range(steps):

        optimizer.zero_grad()

        # build input vector
        x_raw = torch.stack([
            vc,
            fz,
            eps_r,
            eps_a,
            torch.tensor([float(z)], device=device),
            torch.tensor([float(ri)], device=device)
        ]).T

        # normalize input (VERY IMPORTANT)
        x = (x_raw - X_mean) / X_std

        pred = model(x)

        loss = torch.mean((pred - target) ** 2)

        loss.backward()
        optimizer.step()

        # clamp bounds
        with torch.no_grad():
            vc.clamp_(*vc_bounds)
            fz.clamp_(*fz_bounds)
            eps_r.clamp_(*eps_r_bounds)
            eps_a.clamp_(*eps_a_bounds)

        if step % 100 == 0:
            print(f"step {step} | loss {loss.item():.6f}")

    # evaluate final solution
    with torch.no_grad():

        x_final_raw = torch.tensor([[vc.item(),
                                     fz.item(),
                                     eps_r.item(),
                                     eps_a.item(),
                                     z,
                                     ri]], device=device)

        x_final = (x_final_raw - X_mean) / X_std

        pred_final = model(x_final)

        final_loss = torch.mean((pred_final - target) ** 2).item()

        if final_loss < best_loss:

            best_loss = final_loss

            best_solution = {
                "inputs": x_final_raw.cpu().numpy()[0],
                "pred": pred_final.cpu().numpy()[0]
            }


# ==============================
# Print best result
# ==============================

print("\n==============================")
print("BEST SOLUTION FOUND")
print("==============================")

inputs = best_solution["inputs"]
pred = best_solution["pred"]

print("\nOptimized parameters:")
print(f"vc   = {inputs[0]:.3f}")
print(f"fz   = {inputs[1]:.3f}")
print(f"eps_r= {inputs[2]:.5f}")
print(f"eps_a= {inputs[3]:.5f}")
print(f"z    = {int(inputs[4])}")
print(f"ri   = {int(inputs[5])}")

print("\nPredicted surface metrics:")
print(f"Ra = {pred[0]:.4f}")
print(f"Rq = {pred[1]:.4f}")
print(f"Rz = {pred[2]:.4f}")

print(f"\nFinal loss: {best_loss:.6f}")