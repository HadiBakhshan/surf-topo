import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import interp1d


# ==========================================================
# CUTTING EDGE TRAJECTORIES
# ==========================================================

def plot_cutting_edge_trajectories(trajectory_points, plot_3d=False, save_path=None):
    """
    Plot cutting edge trajectories in 2D or 3D.
    
    Parameters
    ----------
    trajectory_points : list of tuples/lists
        Each point: (K, t, x, y, z)
    plot_3d : bool
        If True, plot in 3D; otherwise 2D XY projection.
    save_path : str, optional
        If provided, save the figure to this path instead of showing.
    """

    Ks = sorted(set(p[0] for p in trajectory_points))  # unique cutting edges

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

        if plot_3d:
            ax.plot(x, y, z, label=f'K={K}', color=colors(i))
        else:
            ax.plot(x, y, label=f'K={K}', color=colors(i))

    ax.set_title("Cutting Edge Trajectories")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    if plot_3d:
        ax.set_zlabel('Z')

    ax.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


# ==========================================================
# SURFACE SCATTER
# ==========================================================

def plot_surface_scatter(surface, m, n, elev=30, azim=50, cmap='jet',
                         size=10, fixed_zlim=None, box_aspect=(1, 0.5, 0.1),
                         save_path=None):
    """
    Plot a 3D scatter of the surface points colored by Z coordinate.

    Parameters
    ----------
    surface : np.ndarray
        Shape (N, 3), where N = (m+1)*(n+1)
    m, n : int
        Grid dimensions
    save_path : str, optional
        Path to save figure; if None, display the plot
    """

    X = surface[:, 0].reshape((m + 1, n + 1))
    Y = surface[:, 1].reshape((m + 1, n + 1))
    Z_um = surface[:, 2].reshape((m + 1, n + 1)) * 1000  # mm → µm

    x_flat = X.ravel()
    y_flat = Y.ravel()
    z_flat = Z_um.ravel()

    z_min, z_max = np.nanmin(z_flat), np.nanmax(z_flat)
    norm = plt.Normalize(vmin=z_min, vmax=z_max)
    colors = cm.get_cmap(cmap)(norm(z_flat))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_flat, y_flat, z_flat, c=colors, s=size)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (µm)")

    if fixed_zlim is not None:
        ax.set_zlim(-fixed_zlim, fixed_zlim)
        ax.set_zticks(np.linspace(-fixed_zlim, fixed_zlim, 3))
    else:
        z_margin = 0.05 * (z_max - z_min) if z_max != z_min else 1
        ax.set_zlim(z_min - z_margin, z_max + z_margin)
        ax.set_zticks(np.linspace(z_min, z_max, 3))

    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect(box_aspect)

    cbar_ax = fig.add_axes([0.25, 0.75, 0.25, 0.02])
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(z_flat)
    cbar = fig.colorbar(mappable, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Z (μm)')

    plt.title("3D Scatter of Surface Points Colored by Z (μm)")

    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


# ==========================================================
# CROSS SECTIONS
# ==========================================================

def plot_cross_sections(surface, m, n, y_index=None, x_index=None,
                        band_width=0, smooth_factor=5, save_path=None):
    """
    Plot smoothed cross-sections Z vs X and Z vs Y.

    Parameters
    ----------
    surface : np.ndarray
        Shape (N, 3), where N = (m+1)*(n+1)
    m, n : int
        Grid dimensions
    y_index, x_index : int
        Slice index along Y or X; default middle
    band_width : int
        Number of neighboring points to average
    smooth_factor : int
        Interpolation points per interval
    save_path : str, optional
        Path to save figure; if None, display the plot
    """

    X = surface[:, 0].reshape((m + 1, n + 1))
    Y = surface[:, 1].reshape((m + 1, n + 1))
    Z_mm = surface[:, 2].reshape((m + 1, n + 1))
    Z_um = Z_mm * 1000  # mm → µm

    if y_index is None:
        y_index = (n + 1) // 2
    if x_index is None:
        x_index = (m + 1) // 2

    # --- Cross-section along X (fix Y) ---
    y_start = max(0, y_index - band_width)
    y_end = min(n, y_index + band_width)
    z_band_x = Z_um[:, y_start:y_end + 1]
    z_line_x = np.mean(z_band_x, axis=1)
    x_line = X[:, y_index]

    interp_func_x = interp1d(x_line, z_line_x, kind="cubic")
    x_smooth = np.linspace(x_line.min(), x_line.max(), len(x_line) * smooth_factor)
    z_smooth_x = interp_func_x(x_smooth)

    # --- Cross-section along Y (fix X) ---
    x_start = max(0, x_index - band_width)
    x_end = min(m, x_index + band_width)
    z_band_y = Z_um[x_start:x_end + 1, :]
    z_line_y = np.mean(z_band_y, axis=0)
    y_line = Y[x_index, :]

    interp_func_y = interp1d(y_line, z_line_y, kind="cubic")
    y_smooth = np.linspace(y_line.min(), y_line.max(), len(y_line) * smooth_factor)
    z_smooth_y = interp_func_y(y_smooth)

    # --- Plot ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 5))
    axes[0].plot(x_smooth, z_smooth_x, "b-", linewidth=2)
    axes[0].set_xlabel("X (mm)")
    axes[0].set_ylabel("Z (μm)")
    axes[0].set_title(f"Cross-section along X @ Y={y_index}±{band_width}")
    axes[0].grid(True)

    axes[1].plot(y_smooth, z_smooth_y, "g-", linewidth=2)
    axes[1].set_xlabel("Y (mm)")
    axes[1].set_ylabel("Z (μm)")
    axes[1].set_title(f"Cross-section along Y @ X={x_index}±{band_width}")
    axes[1].grid(True)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
