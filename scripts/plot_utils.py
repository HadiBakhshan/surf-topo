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

# def plot_cutting_edge_trajectories(trajectory_points, plot_3d=False, save_path=None):
#     """
#     Plot cutting edge trajectories in 2D or 3D.
    
#     Parameters
#     ----------
#     trajectory_points : list of tuples/lists
#         Each point: (K, t, x, y, z)
#     plot_3d : bool
#         If True, plot in 3D; otherwise 2D XY projection.
#     save_path : str, optional
#         If provided, save the figure to this path instead of showing.
#     """

#     Ks = sorted(set(p[0] for p in trajectory_points))  # unique cutting edges

#     if plot_3d:
#         fig = plt.figure(figsize=(10, 7))
#         ax = fig.add_subplot(111, projection='3d')
#     else:
#         fig, ax = plt.subplots(figsize=(10, 7))

#     colors = plt.cm.get_cmap('tab10', len(Ks))

#     for i, K in enumerate(Ks):
#         traj_K = [p for p in trajectory_points if p[0] == K]
#         x = [p[2] for p in traj_K]
#         y = [p[3] for p in traj_K]
#         z = [p[4] for p in traj_K]

#         if plot_3d:
#             ax.plot(x, y, z, label=f'K={K}', color=colors(i))
#         else:
#             ax.plot(x, y, label=f'K={K}', color=colors(i))

#     ax.set_title("Cutting Edge Trajectories")
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')

#     if plot_3d:
#         ax.set_zlabel('Z')

#     ax.legend()
#     plt.grid(True)
#     plt.tight_layout()

#     if save_path is not None:
#         plt.savefig(save_path)
#         plt.close()
#     else:
#         plt.show()



def plot_cutting_edge_trajectories(trajectory_points, plot_3d=False, save_path=None):
    # Convert list of dicts to structured arrays
    Ks = sorted(set(p[0] for p in trajectory_points))

    if plot_3d:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.get_cmap('tab10', len(Ks))

    for i, K in enumerate(Ks):
        traj_K = [p for p in trajectory_points if p[0] == K]
        x = [p[2] for p in traj_K]
        y = [p[3] for p in traj_K]
        z = [p[4] for p in traj_K]

        label = f'K={K}'

        if plot_3d:
            ax.plot(x, y, z, label=label, color=colors(i))
        else:
            ax.plot(x, y, label=label, color=colors(i))

    ax.set_title("Cutting Edge Trajectories")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    if plot_3d:
        ax.set_zlabel('Z')

    ax.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show(block=False)
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


# ==========================================================
# SURFACE SCATTER
# ==========================================================

# def plot_surface_scatter(surface, m, n, elev=30, azim=50, cmap='jet',
#                          size=10, fixed_zlim=None, box_aspect=(1, 0.5, 0.1),
#                          save_path=None):
#     """
#     Plot a 3D scatter of the surface points colored by Z coordinate.

#     Parameters
#     ----------
#     surface : np.ndarray
#         Shape (N, 3), where N = (m+1)*(n+1)
#     m, n : int
#         Grid dimensions
#     save_path : str, optional
#         Path to save figure; if None, display the plot
#     """

#     X = surface[:, 0].reshape((m + 1, n + 1))
#     Y = surface[:, 1].reshape((m + 1, n + 1))
#     Z_um = surface[:, 2].reshape((m + 1, n + 1)) * 1000  # mm → µm

#     x_flat = X.ravel()
#     y_flat = Y.ravel()
#     z_flat = Z_um.ravel()

#     z_min, z_max = np.nanmin(z_flat), np.nanmax(z_flat)
#     norm = plt.Normalize(vmin=z_min, vmax=z_max)
#     colors = cm.get_cmap(cmap)(norm(z_flat))

#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(x_flat, y_flat, z_flat, c=colors, s=size)

#     ax.set_xlabel("X (mm)")
#     ax.set_ylabel("Y (mm)")
#     ax.set_zlabel("Z (µm)")

#     if fixed_zlim is not None:
#         ax.set_zlim(-fixed_zlim, fixed_zlim)
#         ax.set_zticks(np.linspace(-fixed_zlim, fixed_zlim, 3))
#     else:
#         z_margin = 0.05 * (z_max - z_min) if z_max != z_min else 1
#         ax.set_zlim(z_min - z_margin, z_max + z_margin)
#         ax.set_zticks(np.linspace(z_min, z_max, 3))

#     ax.view_init(elev=elev, azim=azim)
#     ax.set_box_aspect(box_aspect)

#     cbar_ax = fig.add_axes([0.25, 0.75, 0.25, 0.02])
#     mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
#     mappable.set_array(z_flat)
#     cbar = fig.colorbar(mappable, cax=cbar_ax, orientation='horizontal')
#     cbar.set_label('Z (μm)')

#     plt.title("3D Scatter of Surface Points Colored by Z (μm)")

#     if save_path is not None:
#         plt.savefig(save_path)
#         plt.close()
#     else:
#         plt.show()


def plot_surface_scatter(
    surface,
    m, n,
    elev=30, azim=50,
    cmap='jet',
    size=10,
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
    z_avg = np.nanmean(z_flat)   # mean height in µm
    print(f"Average surface height: {z_avg:.3f} µm")

    norm = plt.Normalize(vmin=z_min, vmax=z_max)
    colors = cm.get_cmap(cmap)(norm(z_flat))

    # -----------------------------
    # Figure (tight layout, no margins)
    # -----------------------------
    fig = plt.figure(figsize=(8, 6))
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
    ax.set_xlabel("X (mm) [Pick-feed dir. \u2192]", labelpad=15)
    ax.set_ylabel("Y (mm) [Feed dir. \u2190]", labelpad=2)
    ax.set_zlabel("Z (μm)", rotation=270, labelpad=0.5)

    # -----------------------------
    # Axis limits
    # -----------------------------
    # ax.set_xlim(np.min(x_flat), np.max(x_flat))
    # ax.set_ylim(np.min(y_flat), np.max(y_flat))

    ax.set_xlim(np.max(x_flat), np.min(x_flat))
    ax.set_ylim(np.max(y_flat), np.min(y_flat))
    ax.set_zlim(-10, 5)

    if fixed_zlim is not None:
        ax.set_zlim(-fixed_zlim, fixed_zlim)
    else:
        z_margin = 0.05 * (z_max - z_min) if z_max != z_min else 1
        ax.set_zlim(z_min - z_margin, z_max + z_margin)

    ax.set_zlim(-10, z_max)
    ax.set_zticks(np.linspace(z_min, z_max, 3))
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


    # Display average height on plot
    ax.text2D(
        0.02, 0.95,
        f"Mean Z = {z_avg:.2f} µm",
        transform=ax.transAxes,
        fontsize=10
    )

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
    cbar.set_ticks(np.linspace(z_min, z_max, 5))
    cbar.ax.tick_params(length=5, direction='in')

    # plt.show()
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()




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
    # cbar_ax = fig.add_axes([0.1, 0.93, 0.8, 0.02])
    # cbar = fig.colorbar(c, cax=cbar_ax, orientation='horizontal')

    # cbar.ax.xaxis.set_ticks_position('top')
    # cbar.ax.xaxis.set_label_position('top')
    # cbar.ax.tick_params(direction='in')
    # cbar.ax.xaxis.set_major_formatter('{:.0f}'.format)
    # cbar.set_label("μm")

    # plt.show()
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()




# ==========================================================
# CROSS SECTIONS
# ==========================================================

# def plot_cross_sections(surface, m, n, y_index=None, x_index=None,
#                         band_width=0, smooth_factor=5, save_path=None):
#     """
#     Plot smoothed cross-sections Z vs X and Z vs Y.

#     Parameters
#     ----------
#     surface : np.ndarray
#         Shape (N, 3), where N = (m+1)*(n+1)
#     m, n : int
#         Grid dimensions
#     y_index, x_index : int
#         Slice index along Y or X; default middle
#     band_width : int
#         Number of neighboring points to average
#     smooth_factor : int
#         Interpolation points per interval
#     save_path : str, optional
#         Path to save figure; if None, display the plot
#     """

#     X = surface[:, 0].reshape((m + 1, n + 1))
#     Y = surface[:, 1].reshape((m + 1, n + 1))
#     Z_mm = surface[:, 2].reshape((m + 1, n + 1))
#     Z_um = Z_mm * 1000  # mm → µm

#     if y_index is None:
#         y_index = (n + 1) // 2
#     if x_index is None:
#         x_index = (m + 1) // 2

#     # --- Cross-section along X (fix Y) ---
#     y_start = max(0, y_index - band_width)
#     y_end = min(n, y_index + band_width)
#     z_band_x = Z_um[:, y_start:y_end + 1]
#     z_line_x = np.mean(z_band_x, axis=1)
#     x_line = X[:, y_index]

#     interp_func_x = interp1d(x_line, z_line_x, kind="cubic")
#     x_smooth = np.linspace(x_line.min(), x_line.max(), len(x_line) * smooth_factor)
#     z_smooth_x = interp_func_x(x_smooth)

#     # --- Cross-section along Y (fix X) ---
#     x_start = max(0, x_index - band_width)
#     x_end = min(m, x_index + band_width)
#     z_band_y = Z_um[x_start:x_end + 1, :]
#     z_line_y = np.mean(z_band_y, axis=0)
#     y_line = Y[x_index, :]

#     interp_func_y = interp1d(y_line, z_line_y, kind="cubic")
#     y_smooth = np.linspace(y_line.min(), y_line.max(), len(y_line) * smooth_factor)
#     z_smooth_y = interp_func_y(y_smooth)

#     # --- Plot ---
#     fig, axes = plt.subplots(2, 1, figsize=(12, 5))
#     axes[0].plot(x_smooth, z_smooth_x, "b-", linewidth=2)
#     axes[0].set_xlabel("X (mm)")
#     axes[0].set_ylabel("Z (μm)")
#     axes[0].set_title(f"Cross-section along X @ Y={y_index}±{band_width}")
#     axes[0].grid(True)

#     axes[1].plot(y_smooth, z_smooth_y, "g-", linewidth=2)
#     axes[1].set_xlabel("Y (mm)")
#     axes[1].set_ylabel("Z (μm)")
#     axes[1].set_title(f"Cross-section along Y @ X={x_index}±{band_width}")
#     axes[1].grid(True)

#     plt.tight_layout()

#     if save_path is not None:
#         plt.savefig(save_path)
#         plt.close()
#     else:
#         plt.show()



# def plot_cross_sections(surface, m, n, y_index=None, x_index=None,
#                         band_width=0, smooth_window=11, poly_order=3,
#                         save_path=None):
#     """
#     Plot smoothed cross-sections Z vs X and Z vs Y using real values
#     (no cubic spline overshoot).

#     Parameters
#     ----------
#     surface : np.ndarray
#         Shape (N, 3), where N = (m+1)*(n+1)
#     m, n : int
#         Grid dimensions
#     y_index, x_index : int
#         Slice index along Y or X; default middle
#     band_width : int
#         Number of neighboring points to average
#     smooth_window : int
#         Window length for Savitzky–Golay filter (must be odd)
#     poly_order : int
#         Polynomial order for Savitzky–Golay filter
#     save_path : str, optional
#         Path to save figure; if None, display the plot
#     """

#     import numpy as np
#     import matplotlib.pyplot as plt
#     from scipy.signal import savgol_filter

#     # -----------------------------
#     # Reshape structured surface
#     # -----------------------------
#     X = surface[:, 0].reshape((m + 1, n + 1))
#     Y = surface[:, 1].reshape((m + 1, n + 1))
#     Z_mm = surface[:, 2].reshape((m + 1, n + 1))
#     Z_um = Z_mm * 1000  # mm → µm

#     if y_index is None:
#         y_index = (n + 1) // 2
#     if x_index is None:
#         x_index = (m + 1) // 2

#     # Ensure valid smoothing window
#     def safe_savgol(z):
#         if len(z) < smooth_window:
#             return z  # too short → no smoothing
#         if smooth_window % 2 == 0:
#             window = smooth_window + 1
#         else:
#             window = smooth_window
#         return savgol_filter(z, window, poly_order)

#     # ==========================================================
#     # --- Cross-section along X (fix Y) ---
#     # ==========================================================
#     y_start = max(0, y_index - band_width)
#     y_end = min(n + 1, y_index + band_width + 1)

#     z_band_x = Z_um[:, y_start:y_end]
#     z_line_x = np.mean(z_band_x, axis=1)   # real averaged values
#     x_line = X[:, y_index]

#     z_smooth_x = safe_savgol(z_line_x)

#     # ==========================================================
#     # --- Cross-section along Y (fix X) ---
#     # ==========================================================
#     x_start = max(0, x_index - band_width)
#     x_end = min(m + 1, x_index + band_width + 1)

#     z_band_y = Z_um[x_start:x_end, :]
#     z_line_y = np.mean(z_band_y, axis=0)   # real averaged values
#     y_line = Y[x_index, :]

#     z_smooth_y = safe_savgol(z_line_y)

#     # ==========================================================
#     # --- Plot ---
#     # ==========================================================
#     fig, axes = plt.subplots(2, 1, figsize=(12, 5))

#     # X profile
#     axes[0].plot(x_line, z_smooth_x, linewidth=2)
#     axes[0].set_xlabel("X (mm)")
#     axes[0].set_ylabel("Z (μm)")
#     axes[0].set_title(f"Cross-section along X @ Y={y_index} ± {band_width}")
#     axes[0].grid(True)

#     # Y profile
#     axes[1].plot(y_line, z_smooth_y, linewidth=2)
#     axes[1].set_xlabel("Y (mm)")
#     axes[1].set_ylabel("Z (μm)")
#     axes[1].set_title(f"Cross-section along Y @ X={x_index} ± {band_width}")
#     axes[1].grid(True)

#     plt.tight_layout()

#     if save_path is not None:
#         plt.savefig(save_path)
#         plt.close()
#     else:
#         plt.show()


def plot_cross_sections(surface, m, n,
                        y_index=None, x_index=None,
                        band_width=0,
                        smooth_window=11, poly_order=3,
                        exp_x_file=r"D:\Codes\surf-topo\cases\exp\A1_1.csv",
                        exp_y_file=r"D:\Codes\surf-topo\cases\exp\A1_2.csv",
                        save_path=None):
    """
    Plot cross-sections Z vs X and Z vs Y with optional experimental
    curves loaded from Excel or CSV files.

    Experimental file format:
        column 1 → horizontal axis
        column 2 → vertical axis (height)

    Parameters
    ----------
    surface : np.ndarray
        Shape (N, 3)
    m, n : int
        Grid dimensions
    exp_x_file : str
        Experimental curve for X section (upper plot)
    exp_y_file : str
        Experimental curve for Y section (lower plot)
    """

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
    # Robust experimental data loader
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
            x = data.iloc[:, 0].values
            z = data.iloc[:, 1].values
            return x, z
        except Exception as e:
            print(f"Error loading experimental file {path}: {e}")
            return None, None

    exp_x, exp_zx = load_profile(exp_x_file)
    exp_y, exp_zy = load_profile(exp_y_file)

    # -----------------------------
    # Plotting
    # -----------------------------
    fig, axes = plt.subplots(2, 1, figsize=(12, 5))

    # X profile
    axes[0].plot(x_line, z_smooth_x, linewidth=2, label="Simulated")
    if exp_x is not None:
        axes[0].plot(exp_x, exp_zx, '--o', markersize=3, linewidth=2, label="Experimental")
    axes[0].set_xlabel("X (mm)")
    axes[0].set_ylabel("Z (μm)")
    axes[0].set_title(f"Cross-section along X @ Y={y_index} ± {band_width}")
    axes[0].grid(True)
    axes[0].legend()

    # Y profile
    axes[1].plot(y_line, z_smooth_y, linewidth=2, label="Simulated")
    if exp_y is not None:
        axes[1].plot(exp_y, exp_zy, '--o', markersize=3, linewidth=2, label="Experimental")
    axes[1].set_xlabel("Y (mm)")
    axes[1].set_ylabel("Z (μm)")
    axes[1].set_title(f"Cross-section along Y @ X={x_index} ± {band_width}")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

