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
    elev=30, azim=50,
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

    # -----------------------------
    # Average height calculation
    # -----------------------------
    # z_avg = np.nanmean(z_flat)   # mean height in µm
    # print(f"Average surface height: {z_avg:.3f} µm")

    # -----------------------------
    # Surface roughness parameters
    # -----------------------------

    # Mean height
    z_mean = np.mean(z_flat)

    # Ra: arithmetic average roughness
    Ra = np.mean(np.abs(z_flat - z_mean))

    # Rq: root mean square roughness
    Rq = np.sqrt(np.mean(z_flat**2))

    # Rz: peak-to-valley height
    Rz = np.max(z_flat) - np.min(z_flat)

    print(f"Ra (average roughness): {Ra:.3f} µm")
    print(f"Rq (RMS roughness): {Rq:.3f} µm")
    print(f"Rz (peak-to-valley): {Rz:.3f} µm")

    norm = plt.Normalize(vmin=z_min, vmax=z_max)
    colors = cm.get_cmap(cmap)(norm(z_flat))

    # -----------------------------
    # Figure (tight layout, no margins)
    # -----------------------------
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Remove outer white margins
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
    # ax.set_zlabel("Z (μm)", rotation=90, labelpad=15, fontsize=20)

    # -----------------------------
    # Axis limits
    # -----------------------------
    # ax.set_xlim(np.min(x_flat), np.max(x_flat))
    # ax.set_ylim(np.min(y_flat), np.max(y_flat))

    ax.set_xlim(np.max(x_flat), np.min(x_flat))
    ax.set_ylim(np.max(y_flat), np.min(y_flat))
    ax.set_zlim(-10, 5)

    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.tick_params(axis='z', labelsize=18)

    if fixed_zlim is not None:
        ax.set_zlim(-fixed_zlim, fixed_zlim)
    else:
        z_margin = 0.05 * (z_max - z_min) if z_max != z_min else 1
        ax.set_zlim(z_min - z_margin, z_max + z_margin)

    ax.set_zlim(-10, z_max)
    ax.set_zticks(np.linspace(-10, 10, 3))
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
    # Compact colorbar (close to plot)
    # -----------------------------
    cbar_ax = fig.add_axes([0.15, 0.62, 0.18, 0.02])
    # [left, bottom, width, height]

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(z_flat)

    cbar = fig.colorbar(mappable, cax=cbar_ax, orientation='horizontal')

    cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    # Move label to RIGHT side
    cbar.ax.set_xlabel("μm", fontsize=9, labelpad=5)
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.set_ticks_position('top')   # move numbers to top
    mid_vals = np.linspace(z_min, z_max, 6)[1:-1]
    cbar.set_ticks(mid_vals)
    cbar.ax.tick_params(length=5, direction='in')

    ax.xaxis.set_tick_params(pad=14)
    ax.yaxis.set_tick_params(pad=18)
    ax.zaxis.set_tick_params(pad=10)

    cbar.ax.set_xlabel("μm", fontsize=16, labelpad=7)
    cbar.ax.tick_params(labelsize=18)

    # plt.show()
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
    save_path=None):

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

    # 🔹 Swap X and Y here
    c = ax.pcolormesh(Y, X, Z_um,
                      cmap=cmap,
                      shading='auto',
                      vmin=vmin,
                      vmax=vmax)

    # 🔹 Swap labels
    ax.set_xlabel("Y (mm)")
    ax.set_ylabel("X (mm)", rotation = 270, labelpad=15)

    ax.set_xlim(np.min(Y), np.max(Y))   # Y normal (left → right)
    ax.set_ylim(np.max(X), np.min(X))   # X inverted (top → down)


    ax.set_aspect('equal')

    # --- Move horizontal axis to top ---
    ax.xaxis.set_ticks_position('top')
    # ax.xaxis.set_label_position('top')

    # --- Move vertical axis to right ---
    # ax.yaxis.set_ticks_position('right')
    ax.yaxis.set_label_position('right')


    x_ticks = np.linspace(np.min(Y), np.max(Y), 3)
    y_ticks = np.linspace(np.min(X), np.max(X), 5)

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # Optional: make ticks point inward
    ax.tick_params(direction='in')

    # Colorbar
    cbar_ax = fig.add_axes([0.1, 0.93, 0.8, 0.02])
    cbar = fig.colorbar(c, cax=cbar_ax, orientation='horizontal')

    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.tick_params(direction='in')
    cbar.ax.xaxis.set_major_formatter('{:.0f}'.format)
    cbar.set_label("μm")

    plt.show()
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


# ==========================================================
# ROUGHNESS PLOT
# ==========================================================


def plot_cross_sections(surface, m, n,
                        y_index=None, x_index=None,
                        band_width=0,
                        smooth_window=11, poly_order=3,
                        exp_x_file=r"D:\Codes\surf-topo\cases\exp\A3_1.csv",
                        exp_y_file=r"D:\Codes\surf-topo\cases\exp\A3_2.csv",
                        save_path=None):

    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import pandas as pd
    from scipy.signal import savgol_filter

    # -----------------------------
    # Reshape surface
    # -----------------------------
    X = surface[:, 0].reshape((m + 1, n + 1))
    Y = surface[:, 1].reshape((m + 1, n + 1))
    Z_um = surface[:, 2].reshape((m + 1, n + 1)) * 1000  # mm → µm

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
    # Integral-based mean function
    # -----------------------------
    def integral_mean(x, z):
        """Compute mean height using integral definition."""
        if x is None or z is None:
            return None
        length = x.max() - x.min()
        if length == 0:
            return np.mean(z)
        area = np.trapz(z, x)
        return area / length

    # -----------------------------
    # Cross-section along X
    # -----------------------------
    y_start = max(0, y_index - band_width)
    y_end = min(n + 1, y_index + band_width + 1)
    z_band_x = Z_um[:, y_start:y_end]
    z_line_x = np.mean(z_band_x, axis=1)
    x_line = X[:, y_index]
    z_smooth_x = safe_savgol(z_line_x)

    # -----------------------------
    # Cross-section along Y
    # -----------------------------
    x_start = max(0, x_index - band_width)
    x_end = min(m + 1, x_index + band_width + 1)
    z_band_y = Z_um[x_start:x_end, :]
    z_line_y = np.mean(z_band_y, axis=0)
    y_line = Y[x_index, :]
    z_smooth_y = safe_savgol(z_line_y)

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
    # Compute integral-based means
    # -----------------------------
    mean_pred_x = integral_mean(x_line, z_smooth_x)
    mean_pred_y = integral_mean(y_line, z_smooth_y)

    mean_exp_x = integral_mean(exp_x, exp_zx) if exp_x is not None else None
    mean_exp_y = integral_mean(exp_y, exp_zy) if exp_y is not None else None


    print("\n===== CROSS-SECTION MEAN HEIGHTS (µm) =====")
    
    # if mean_exp_x is not None:
    #     print(f"Experimental X-section mean: {mean_exp_x:.3f}")

    print(f"Prediction Y-section mean (Feed direction): {mean_pred_y:.3f}")

    print(f"Prediction X-section mean (pick-feed direction): {mean_pred_x:.3f}")
    # if mean_exp_y is not None:
    #     print(f"Experimental Y-section mean: {mean_exp_y:.3f}")
    print("==========================================\n")

    # -----------------------------
    # Plotting
    # -----------------------------
    fig, axes = plt.subplots(2, 1, figsize=(12, 5))

    def apply_axis_rules(ax, exp_axis):
        if exp_axis is not None:
            ax.set_xlim(np.min(exp_axis), np.max(exp_axis))
        ax.set_ylim(-10, 10)

        ax.legend(loc='lower center',
                  bbox_to_anchor=(0.5, 0.01),
                  ncol=2,
                  frameon=False)

    def draw_mean_line(ax, mean_val, label):
        ax.axhline(mean_val, linewidth=1.2)
        ax.text(ax.get_xlim()[1],
                mean_val,
                f"{label}: {mean_val:.2f}",
                va='center',
                ha='right',
                fontsize=9)

    # ---- X profile ----
    axes[0].plot(x_line, z_smooth_x, linewidth=2, label="Prediction")

    if exp_x is not None:
        axes[0].plot(exp_x, exp_zx, '--o', markersize=3,
                     linewidth=2, label="Experimental")

    axes[0].set_xlabel("X (mm)")
    axes[0].set_ylabel("Z (μm)")
    axes[0].grid(False)
    apply_axis_rules(axes[0], exp_x)

    # draw_mean_line(axes[0], mean_pred_x, "Mean Pred")
    # if mean_exp_x is not None:
    #     draw_mean_line(axes[0], mean_exp_x, "Mean Exp")

    # ---- Y profile ----
    axes[1].plot(y_line, z_smooth_y, linewidth=2, label="Prediction")

    if exp_y is not None:
        axes[1].plot(exp_y, exp_zy, '--o', markersize=3,
                     linewidth=2, label="Experimental")

    axes[1].set_xlabel("Y (mm)")
    axes[1].set_ylabel("Z (μm)")
    axes[1].grid(False)
    apply_axis_rules(axes[1], exp_y)

    # draw_mean_line(axes[1], mean_pred_y, "Mean Pred")
    # if mean_exp_y is not None:
    #     draw_mean_line(axes[1], mean_exp_y, "Mean Exp")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
