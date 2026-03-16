import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import interp1d
from matplotlib.ticker import FormatStrFormatter
import openpyxl


# ==========================================================
# CUTTING EDGE TRAJECTORIES
# ==========================================================

def plot_cutting_edge_trajectories(trajectory_points, plot_3d=False, save_path=None):
    import matplotlib.pyplot as plt

    # Extract unique K values
    Ks = sorted(set(p[0] for p in trajectory_points))

    # Create figure
    if plot_3d:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = plt.subplots(figsize=(10, 7))

    # Custom colors: sky blue & dark orange
    # colors = ['#1E90FF', '#FF8C00']
    # colors = ["#0A91E0", "#ED0C1E", "#3EC347", "#F68D38"]
    colors = ["#0A91E0", "#F68D38", "#3EC347", "#ED0C1E"]
    

    # Plot each trajectory
    for i, K in enumerate(Ks):
        traj_K = [p for p in trajectory_points if p[0] == K]

        x = [p[2] for p in traj_K]
        y = [p[3] for p in traj_K]
        z = [p[4] for p in traj_K]

        label = f'K={K}'
        color = colors[i % len(colors)]

        if plot_3d:
            ax.plot(x, y, z, label=label, color=color)
        else:
            ax.plot(x, y, label=label, color=color)

    # Axis labels
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    if plot_3d:
        ax.set_zlabel('Z')

    ax.legend()
    plt.tight_layout()

    # Save or show
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


# ==========================================================
# 3D TOPOGRAPHY
# ==========================================================


def plot_surface_scatter(
    surface,
    m, n,
    elev=45, azim=50,
    cmap='jet',
    size=15,
    fixed_zlim=None,
    box_aspect=(1, 0.5, 0.1),
    save_path=None):
    """
    Plot a 3D scatter of structured (N, 3) surface data,
    colored by Z values.
    """

    # -----------------------------
    # Extract and reshape
    # -----------------------------
    X = surface[:, 0].reshape((m + 1, n + 1))
    Y = surface[:, 1].reshape((m + 1, n + 1))
    Z_mm = surface[:, 2].reshape((m + 1, n + 1))
    Z_um = Z_mm * 1000

    x_flat = X.ravel()
    y_flat = Y.ravel()
    z_flat = Z_um.ravel()

    z_min, z_max = np.nanmin(z_flat), np.nanmax(z_flat)

    # Mean height
    z_mean = np.mean(z_flat)

    # Centered heights
    z_centered = z_flat - z_mean

    # Sa: arithmetic average roughness
    Sa = np.mean(np.abs(z_centered))

    # Sq: root mean square roughness
    Sq = np.sqrt(np.mean(z_centered**2))

    # Sz: peak-to-valley height
    Sz = np.max(z_flat) - np.min(z_flat)

    # Sp: maximum peak height
    Sp = np.max(z_centered)

    # Sv: maximum valley depth
    Sv = -np.min(z_centered)

    # Ssk: skewness
    Ssk = np.mean(z_centered**3) / (Sq**3)

    # Sku: kurtosis
    Sku = np.mean(z_centered**4) / (Sq**4)

    print(f"Sa (average roughness): {Sa:.3f} µm")
    print(f"Sq (RMS roughness): {Sq:.3f} µm")
    print(f"Sz (peak-to-valley): {Sz:.3f} µm")

    print(f"Sp (max peak height): {Sp:.3f} µm")
    print(f"Sv (max valley depth): {Sv:.3f} µm")
    print(f"Ssk (skewness): {Ssk:.3f}")
    print(f"Sku (kurtosis): {Sku:.3f}")

        
    margin = 6

    vmin = z_min
    vmax = z_max + margin

    norm = plt.Normalize(vmin, vmax)
    colors = cm.get_cmap(cmap)(norm(z_flat))


    # -----------------------------
    # Figure (tight layout, no margins)
    # -----------------------------
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # -----------------------------
    # Scatter
    # -----------------------------
    ax.scatter(x_flat, y_flat, z_flat, c=colors, s=size)

    # -----------------------------
    # Axis labels
    # -----------------------------
    ax.set_xlabel("X (mm) [Pick-feed dir. \u2192]", labelpad=40, fontsize=20)
    ax.set_ylabel("Y (mm) [Feed dir. \u2190]", labelpad=25, fontsize=20)

    # -----------------------------
    # Axis limits
    # -----------------------------
    ax.set_xlim(np.max(x_flat), np.min(x_flat))
    ax.set_ylim(np.max(y_flat), np.min(y_flat))
    ax.set_zlim(-10, 5)

    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.tick_params(axis='z', labelsize=15)

    if fixed_zlim is not None:
        ax.set_zlim(-fixed_zlim, fixed_zlim)
    else:
        z_margin = 0.05 * (z_max - z_min) if z_max != z_min else 1
        ax.set_zlim(z_min - z_margin, z_max + z_margin)

    ax.set_zlim(-10, vmax)
    ax.set_zticks(np.linspace(-10, 10, 5))
    ax.zaxis.set_major_formatter('{:.0f}'.format)

    # -----------------------------
    # Remove grid and panes
    # -----------------------------
    ax.grid(False)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')

    # -----------------------------
    # View and aspect
    # -----------------------------
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect(box_aspect)

    # -----------------------------
    # Vertical colorbar on right
    # -----------------------------
    cbar_ax = fig.add_axes([0.92, 0.15, 0.022, 0.7])
    # [left, bottom, width, height]

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(z_flat)

    cbar = fig.colorbar(mappable, cax=cbar_ax, orientation='vertical')

    # Evenly spaced ticks along the vertical bar
    ticks = np.linspace(vmin, vmax, 6)[1:-1]  # exclude min and max if desired
    cbar.set_ticks(ticks)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    # Set label
    cbar.set_label("μm", fontsize=16, labelpad=10)

    # Tick parameters
    cbar.ax.tick_params(
        labelsize=16,
        length=6,
        width=1,
        direction='in'
    )

    # -----------------------------
    # Tick spacing
    # -----------------------------
    ax.xaxis.set_tick_params(pad=14)
    ax.yaxis.set_tick_params(pad=18)
    ax.zaxis.set_tick_params(pad=15)

    # -----------------------------
    # Save / show
    # -----------------------------
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


# ==========================================================
# 2D TOPOGRAPHY
# ==========================================================

def plot_surface_topview(
    surface,
    m, n,
    cmap='jet',
    fixed_zlim=None,
    save_path=None
):

    X = surface[:, 0].reshape((m + 1, n + 1))
    Y = surface[:, 1].reshape((m + 1, n + 1))
    Z_mm = surface[:, 2].reshape((m + 1, n + 1))
    Z_um = Z_mm * 1000

    z_min, z_max = np.nanmin(Z_um), np.nanmax(Z_um)

    if fixed_zlim is not None:
        vmin, vmax = -fixed_zlim, fixed_zlim
    else:
        vmin, vmax = z_min, z_max

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    margin = 6
    vmax = z_max + margin

    # 🔹 Swap X and Y here
    c = ax.pcolormesh(
        Y, X, Z_um,
        cmap=cmap,
        shading='auto',
        vmin=vmin,
        vmax=vmax
    )

    # 🔹 Swap labels
    ax.set_xlabel("Y (mm)")
    ax.set_ylabel("X (mm)", rotation=270, labelpad=15)

    ax.set_xlim(np.min(Y), np.max(Y))   # Y normal (left → right)
    ax.set_ylim(np.max(X), np.min(X))   # X inverted (top → down)

    ax.set_aspect('equal')

    # --- Move horizontal axis to top ---
    ax.xaxis.set_ticks_position('top')

    # --- Move vertical axis to right ---
    ax.yaxis.set_label_position('right')

    # --- Main numbers only ---
    # You can pick e.g., 3 ticks for X, 3 for Y
    x_ticks = np.linspace(np.min(Y), np.max(Y), 3)
    y_ticks = np.linspace(np.min(X), np.max(X), 3)

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # Optional: make ticks point inward
    ax.tick_params(direction='in', labelsize=12)

    # -----------------------------
    # Vertical colorbar on right
    # -----------------------------
    cbar_ax = fig.add_axes([0.88, 0.1, 0.022, 0.7])
    # [left, bottom, width, height] → right-hand side, tall vertical bar

    cbar = fig.colorbar(c, cax=cbar_ax, orientation='vertical')

    # You can control which numbers appear on the colorbar
    # For example, show only main ticks (like 0, 2, 4)
    num_ticks = 5  # choose how many tick labels
    ticks = np.linspace(vmin, vmax, 6)[1:-1]  # exclude min and max if desired
    cbar.set_ticks(ticks)
    cbar.ax.yaxis.set_major_formatter('{:.0f}'.format)

    cbar.ax.tick_params(labelsize=12, direction='in')
    cbar.set_label("μm", fontsize=14, labelpad=10)

    # -----------------------------
    # Show or save
    # -----------------------------
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ==========================================================
# ROUGHNESS PLOT
# ==========================================================

import os
import pandas as pd
from scipy.signal import savgol_filter

def plot_cross_sections(surface, m, n,
                        y_index=None, x_index=None,
                        band_width=0,
                        smooth_window=11, poly_order=3,
                        exp_x_file=r"D:\Codes\surf-topo\cases\exp\A3_1.csv",
                        exp_y_file=r"D:\Codes\surf-topo\cases\exp\A3_2.csv",
                        save_path=None):

    # -----------------------------
    # Reshape surface
    # -----------------------------
    X = surface[:, 0].reshape((m + 1, n + 1))
    Y = surface[:, 1].reshape((m + 1, n + 1))
    Z_um = surface[:, 2].reshape((m + 1, n + 1)) * 1000

    if y_index is None:
        y_index = (n + 1) // 2
    if x_index is None:
        x_index = (m + 1) // 2

    # -----------------------------
    # Safe smoothing
    # -----------------------------
    def safe_savgol(z):
        if len(z) < smooth_window:
            return z
        window = smooth_window if smooth_window % 2 else smooth_window + 1
        return savgol_filter(z, window, poly_order)

    # -----------------------------
    # Cross-section along X
    # -----------------------------
    y_start = max(0, y_index - band_width)
    y_end = min(n + 1, y_index + band_width + 1)

    z_band_x = Z_um[:, y_start:y_end]
    z_line_x = np.mean(z_band_x, axis=1)
    x_line = X[:, y_index]

    z_smooth_x = safe_savgol(z_line_x)
    z_center_x = z_smooth_x - np.mean(z_smooth_x)

    # -----------------------------
    # Cross-section along Y
    # -----------------------------
    x_start = max(0, x_index - band_width)
    x_end = min(m + 1, x_index + band_width + 1)

    z_band_y = Z_um[x_start:x_end, :]
    z_line_y = np.mean(z_band_y, axis=0)
    y_line = Y[x_index, :]

    z_smooth_y = safe_savgol(z_line_y)
    z_center_y = z_smooth_y - np.mean(z_smooth_y)

    # -----------------------------
    # Profile Roughness Metrics
    # -----------------------------
    def profile_metrics(z):

        Ra = np.mean(np.abs(z))
        Rq = np.sqrt(np.mean(z**2))

        Rp = np.max(z)
        Rv = -np.min(z)

        Rt = Rp + Rv
        Rz = Rt   # simplified definition for a single profile

        return Ra, Rq, Rt, Rz, Rp, Rv

    Ra_x, Rq_x, Rt_x, Rz_x, Rp_x, Rv_x = profile_metrics(z_center_x)
    Ra_y, Rq_y, Rt_y, Rz_y, Rp_y, Rv_y = profile_metrics(z_center_y)

    print("\n===== PROFILE ROUGHNESS PARAMETERS (µm) =====")

    print("\nX-direction profile (Pick-feed direction)")
    print(f"Ra = {Ra_x:.3f}")
    print(f"Rq = {Rq_x:.3f}")
    print(f"Rt = {Rt_x:.3f}")
    print(f"Rz = {Rz_x:.3f}")
    print(f"Rp = {Rp_x:.3f}")
    print(f"Rv = {Rv_x:.3f}")

    print("\nY-direction profile (Feed direction)")
    print(f"Ra = {Ra_y:.3f}")
    print(f"Rq = {Rq_y:.3f}")
    print(f"Rt = {Rt_y:.3f}")
    print(f"Rz = {Rz_y:.3f}")
    print(f"Rp = {Rp_y:.3f}")
    print(f"Rv = {Rv_y:.3f}")

    print("\n============================================\n")

    # -----------------------------
    # Experimental data loader
    # -----------------------------
    def load_profile(path):

        if not os.path.exists(path):
            print(f"Experimental file not found: {path}")
            return None, None

        try:
            if path.lower().endswith(".csv"):
                data = pd.read_csv(path, header=None)
            else:
                data = pd.read_excel(path, header=None, engine="openpyxl")

            return data.iloc[:, 0].values, data.iloc[:, 1].values

        except Exception as e:
            print(f"Error loading experimental file {path}: {e}")
            return None, None


    exp_x, exp_zx = load_profile(exp_x_file)
    exp_y, exp_zy = load_profile(exp_y_file)

    # -----------------------------
    # Plotting
    # -----------------------------
    fig, axes = plt.subplots(2, 1, figsize=(12, 5))

    # ---- X profile ----
    axes[0].plot(x_line, z_smooth_x, linewidth=2, label="Prediction")

    # Experimental curve (disabled)
    # if exp_x is not None:
    #     axes[0].plot(exp_x, exp_zx, '--o', markersize=3, linewidth=2, label="Experimental")

    axes[0].set_xlabel("X (mm)")
    axes[0].set_ylabel("Z (µm)")
    axes[0].grid(False)

    axes[0].set_xticks(np.linspace(x_line.min(), x_line.max(), 3))
    axes[0].set_yticks(np.linspace(z_smooth_x.min(), z_smooth_x.max(), 5))

    axes[0].legend(frameon=False)

    # ---- Y profile ----
    axes[1].plot(y_line, z_smooth_y, linewidth=2, label="Prediction")

    # Experimental curve (disabled)
    # if exp_y is not None:
    #     axes[1].plot(exp_y, exp_zy, '--o', markersize=3, linewidth=2, label="Experimental")

    axes[1].set_xlabel("Y (mm)")
    axes[1].set_ylabel("Z (µm)")
    axes[1].grid(False)

    axes[1].set_xticks(np.linspace(y_line.min(), y_line.max(), 3))
    axes[1].set_yticks(np.linspace(z_smooth_y.min(), z_smooth_y.max(), 5))

    axes[1].legend(frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()