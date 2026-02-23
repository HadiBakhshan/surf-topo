import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from matplotlib import cm
import time


###########################################################################
#                     classes: Tool, Simulation
###########################################################################



class Tool:
    def __init__(self, gama_f, gama_p, D, eps_r, eps_a, phi, omega, v_f, x_0, y_0, z_0):
        # self.R = R              # cutting_edge_radius
        self.gama_f = gama_f      # radial rake angle of the tool
        self.gama_p = gama_p      # axial rake angle of the tool
        self.D = D                # cutting diameter of the tool
        self.eps_r = eps_r        # radial installation error of inserts
        self.eps_a = eps_a        # axial installation error of inserts
        self.phi = phi            # initial phase angle of the tool
        self.omega = omega        # rotational angular velocity of the tool
        self.v_f = v_f            # feed speed of the tool
        self.fz = fz              # feed per tooth
        # coordinates of the tool in the workpiece coordinate sistem
        self.x_0 = x_0
        self.y_0 = y_0
        self.z_0 = z_0   # lowest position same as cutting-edge position

    def transformation_ct(self,K):  # [cutting edge --> tool]: T_ct

        # Convert degrees → radians
        gamma_f = np.radians(self.gama_f)
        gamma_p = np.radians(self.gama_p)

        T_ct = np.zeros((4, 4))
        T_ct[0, 0] = np.cos(gamma_f)
        T_ct[0, 1] = np.sin(gamma_f) * np.cos(gamma_p)
        T_ct[0, 2] = np.sin(gamma_f) * np.sin(gamma_p)
        T_ct[0, 3] = self.D/2 + (K - 1) * self.eps_r

        T_ct[1, 0] = -np.sin(gamma_f)
        T_ct[1, 1] = np.cos(gamma_f) * np.cos(gamma_p)
        T_ct[1, 2] = np.cos(gamma_f) * np.sin(gamma_p)
        T_ct[1, 3] = 0

        T_ct[2, 0] = 0
        T_ct[2, 1] = -np.sin(gamma_p)
        T_ct[2, 2] = np.cos(gamma_p)
        T_ct[2, 3] = (K - 1) * self.eps_a

        T_ct[3, 3] = 1
        return T_ct


    def transformation_ts(self, K, z_n, t):   # [tool --> spindle]: T_ts

        # Convert phi (deg) → radians before combining with 2π terms
        phi_rad = np.radians(self.phi)

        var1 = phi_rad + ((2*np.pi) * (K - 1)) / z_n - self.omega * t

        T_ts = np.zeros((4, 4))
        T_ts[0, 0] = np.cos(var1)
        T_ts[0, 1] = np.sin(var1)
        T_ts[1, 0] = -np.sin(var1)
        T_ts[1, 1] = np.cos(var1)
        T_ts[2, 2] = 1
        T_ts[3, 3] = 1
        return T_ts


    def transformation_sw(self, t):  # [spindle --> workpiece]: T_sw
        T_sw = np.eye(4)
        T_sw[0, 3] = self.x_0
        T_sw[1, 3] = self.y_0 + self.v_f * t
        T_sw[2, 3] = self.z_0
        return T_sw


    def build_final_transformation_matrix(self, K, z_n, t):

        '''
        T_ct = self.tool.transformation_ct(K)
        T_ts = self.tool.transformation_ts(K, z_n, t)
        T_sw = self.tool.transformation_sw(t)
        return (T_sw @ (T_ts @ T_ct))
        '''

        # Convert degrees → radians
        gamma_f = np.radians(self.gama_f)
        gamma_p = np.radians(self.gama_p)

        # Convert phi (deg) → radians before combining with 2π terms
        var1 = np.radians(self.phi) + ((2 * np.pi) * (K - 1)) / z_n - self.omega * t

        # Precompute trigonometric values
        c_p = np.cos(gamma_p)
        s_p = np.sin(gamma_p)

        # Use trigonometric identities
        cos_diff = np.cos(gamma_f - var1)
        sin_diff = np.sin(gamma_f - var1)

        # Precompute translations
        delta_r = (self.D / 2) + (K - 1) * self.eps_r
        delta_a = (K - 1) * self.eps_a

        # Compute matrix entries directly
        T_final = np.eye(4)

        # First row
        T_final[0, 0] = cos_diff
        T_final[0, 1] = cos_diff * c_p
        T_final[0, 2] = cos_diff * s_p
        T_final[0, 3] = cos_diff * delta_r + sin_diff * delta_a * s_p + self.x_0

        # Second row
        T_final[1, 0] = sin_diff
        T_final[1, 1] = sin_diff * c_p
        T_final[1, 2] = sin_diff * s_p
        T_final[1, 3] = sin_diff * delta_r + cos_diff * delta_a * s_p + self.y_0 + self.v_f * t

        # Third row
        T_final[2, 0] = 0
        T_final[2, 1] = -s_p
        T_final[2, 2] = c_p
        T_final[2, 3] = self.z_0 + delta_a * c_p

        return T_final



class Simulation:
    def __init__(self, tool,
                 d_x, m, d_y, n, z_levels,
                 R, edge_points, z_n,
                 t_total, delta_t):
        self.tool = tool

        # Workpiece parameters
        self.d_x = d_x
        self.m = m
        self.d_y = d_y
        self.n = n
        self.z_levels = z_levels  # assume scalar (flat top surface)
        self.z_n = z_n            # number of the tool teeth

        # Tool parameters
        self.R = R
        self.edge_points = edge_points

        # Time parameters
        self.t_total = t_total
        self.delta_t = delta_t


    def run(self):
        """Run the simulation and return the surface as an (N, 3) point cloud."""
        tolerance = 1e-6

        # --- Discretize Workpiece (grid of points) ---
        x = np.linspace(0, self.d_x, self.m + 1)
        y = np.linspace(0, self.d_y, self.n + 1)
        delta_x = self.d_x / self.m
        delta_y = self.d_y / self.n
        z_level = self.z_levels

        # Create 2D grid of shape (m+1, n+1)
        Z = np.full((self.m + 1, self.n + 1), z_level, dtype=np.float32)

        #print('grid', Z)

        # --- Discretize Tool Edge (1D along X) ---
        l = np.linspace(-self.R, self.R, self.edge_points)
        sqrt_term = np.sqrt(np.clip(self.R**2 - l**2, 0, None))
        tool_points = np.stack([l, np.zeros_like(l), self.R - sqrt_term, np.ones_like(l)], axis=1)  # shape: (edge_points, 4)

        #print('tool', tool_points)

        # --- Discretize Time ---
        times = np.linspace(0, self.t_total, int(self.t_total / self.delta_t) + 1)

        # --- Store cutting edges trajectories ---
        trajectory_points = []  # For storing (x, y, z, t, K)

        for t in times:
            # --- Track modification ---
            Z_modified = np.zeros((self.m + 1, self.n + 1), dtype=bool)

            # Loop over all K cutting-edges (K from 1 to zn inclusive)
            for K in range(1, self.z_n + 1):
                T_matrix = self.tool.build_final_transformation_matrix(K,self.z_n,t)

                transformed_points = (T_matrix @ tool_points.T).T  # shape: (edge_points, 4)

                # Extract coordinates
                x_vals = transformed_points[:, 0]
                y_vals = transformed_points[:, 1]
                z_vals = transformed_points[:, 2]

                min_z_idx = np.argmin(transformed_points[:, 2])
                min_point = transformed_points[min_z_idx]

                trajectory_points.append({
                    'K': K,
                    't': t,
                    'x': min_point[0],
                    'y': min_point[1],
                    'z': min_point[2]
                })

                # Filter: keep points within grid and below z_level
                inside_mask = (
                    (x_vals >= -delta_x) & (x_vals <= self.d_x + delta_x) &
                    (y_vals >= -delta_y) & (y_vals <= self.d_y + delta_y) &
                    (z_vals < z_level + tolerance)
                )
                transformed_points = transformed_points[inside_mask]

                # Skip if no valid points remain
                if transformed_points.shape[0] == 0:
                    continue


                x_idx = ((transformed_points[:, 0]) / delta_x).astype(int)
                y_idx = ((transformed_points[:, 1]) / delta_y).astype(int)

                x_idx = np.clip(x_idx, 0, self.m)
                y_idx = np.clip(y_idx, 0, self.n)

                for i in range(len(transformed_points)):
                    xi = x_idx[i]
                    yi = y_idx[i]
                    z_tool = transformed_points[i, 2]

                    # Check if tool point is within this cell (footprint)
                    if not Z_modified[xi, yi] and z_tool < Z[xi, yi]:
                        Z[xi, yi] = z_tool
                        Z_modified[xi, yi] = True

        # --- Convert Grid to Point Cloud ---
        xx, yy = np.meshgrid(x, y, indexing='ij')  # shape: (m+1, n+1)
        surface = np.stack([xx.ravel(), yy.ravel(), Z.ravel()], axis=1)  # shape: ((m+1)*(n+1), 3)

        return surface, trajectory_points



###########################################################################
#                     plotting functions
###########################################################################



def plot_cutting_edge_trajectories(trajectory_points, plot_3d=False):
    # Convert list of dicts to structured arrays
    Ks = sorted(set(p['K'] for p in trajectory_points))

    if plot_3d:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.get_cmap('tab10', len(Ks))

    for i, K in enumerate(Ks):
        traj_K = [p for p in trajectory_points if p['K'] == K]
        x = [p['x'] for p in traj_K]
        y = [p['y'] for p in traj_K]
        z = [p['z'] for p in traj_K]

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
    plt.show(block=False)



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm



def plot_surface_scatter(
    surface,
    m, n,
    elev=30, azim=50,       # Changed view angle
    cmap='jet',
    size=10,
    fixed_zlim=None,
    box_aspect=(1, 0.5, 0.1)
):
    """
    Plot a 3D scatter of the structured (N, 3) surface data, colored by Z values.
    """
    # Extract and reshape
    X = surface[:, 0].reshape((m + 1, n + 1))
    Y = surface[:, 1].reshape((m + 1, n + 1))
    Z_mm = surface[:, 2].reshape((m + 1, n + 1))
    Z_um = Z_mm * 1000

    x_flat = X.ravel()
    y_flat = Y.ravel()
    z_flat = Z_um.ravel()

    z_min, z_max = np.nanmin(z_flat), np.nanmax(z_flat)
    norm = plt.Normalize(vmin=z_min, vmax=z_max)
    colors = cm.get_cmap(cmap)(norm(z_flat))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x_flat, y_flat, z_flat, c=colors, s=size)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (μm)")

    # Z-limits
    if fixed_zlim is not None:
        ax.set_zlim(-fixed_zlim, fixed_zlim)
        ax.set_zticks(np.linspace(-fixed_zlim, fixed_zlim, 3))
    else:
        z_margin = 0.05 * (z_max - z_min) if z_max != z_min else 1
        ax.set_zlim(z_min - z_margin, z_max + z_margin)
        ax.set_zticks(np.linspace(z_min, z_max, 3))

    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect(box_aspect)

    # Horizontal colorbar at top-left
    cbar_ax = fig.add_axes([0.25, 0.75, 0.25, 0.02])  # [left, bottom, width, height]
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(z_flat)
    cbar = fig.colorbar(mappable, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Z (μm)')

    plt.title("3D Scatter of Surface Points Colored by Z (μm)")
    plt.show()


import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d



def plot_cross_sections(surface, m, n,
                        y_index=None, x_index=None,
                        band_width=0, smooth_factor=5):
    """
    Plot smoothed cross-sections Z vs X and Z vs Y in two subplots.

    Parameters
    ----------
    surface : np.ndarray
        Array of shape (N, 3), where N = (m+1)*(n+1).
    m, n : int
        Grid dimensions used in the simulation.
    y_index : int, optional
        Index along Y-axis to slice (default = middle).
    x_index : int, optional
        Index along X-axis to slice (default = middle).
    band_width : int
        Number of indices above/below slice index to average.
    smooth_factor : int
        Number of interpolation points per original interval (higher = smoother).
    """
    # Reshape surface into structured grid
    X = surface[:, 0].reshape((m + 1, n + 1))
    Y = surface[:, 1].reshape((m + 1, n + 1))
    Z_mm = surface[:, 2].reshape((m + 1, n + 1))
    Z_um = Z_mm * 1000  # convert mm → microns

    # Defaults to middle slices
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

    # --- Plot in two subplots ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 5))

    # Left: Z vs X
    axes[0].plot(x_smooth, z_smooth_x, "b-", linewidth=2)
    axes[0].set_xlabel("X (mm)")
    axes[0].set_ylabel("Z (μm)")
    axes[0].set_title(f"Cross-section along X @ Y={y_index}±{band_width}")
    axes[0].grid(True)

    # Right: Z vs Y
    axes[1].plot(y_smooth, z_smooth_y, "g-", linewidth=2)
    axes[1].set_xlabel("Y (mm)")
    axes[1].set_ylabel("Z (μm)")
    axes[1].set_title(f"Cross-section along Y @ X={x_index}±{band_width}")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()



###########################################################################
#                     main function to run
###########################################################################





if __name__ == "__main__":

    start = time.perf_counter()
    print('Simulation Start')
    #cutting parameters:
    vc = 50                                         # cutting speed m/min
    Td = 10                                           # tool diameter mm
    fz = 0.4                                          # feed per tooth mm/tooth
    z = 2                                             # number of teeth / number of cutting edges
    ap = 0.4                                          # axial depth of the cut mm
    gama_f = 0.6
    gama_p=0.0

    ws = (vc / (Td * 0.001 * np.pi))                  # spindle speed (RPM)
    vf = fz * z * ws / 60                             # feed speed mm/s
    print('Spindle velocity:', round(ws,2), 'RPM')
    print('Feed speed:', round(vf,2), 'mm/s')

    w = ws * np.pi / 30;                              # spindle speed (rad/s)


    tool = Tool(gama_f, gama_p, D=Td, eps_r=-0.026, eps_a=0.009,
                phi=90, omega=w, v_f=vf, x_0=5, y_0=-5, z_0=0)


    #grid dimensions:
    grid =200
    Length = 5
    grid_m= grid*2                  # x grid
    Lx = Length*2                   # total workpice x length
    grid_n= grid                    # y grid
    Ly = Length                     # total workpice y length
    grid_t = max(grid_m,grid_n)*2   # tool grid
    ri = 5                          # insert radius


    total_time = (1.5*Td+Ly)/vf     # total time
    print('Total process time:', round(total_time, 2), 's')

    sim = Simulation(tool, d_x=Lx, m=grid_m, d_y=Ly, n=grid_n, z_levels=ap, R=ri,
                     edge_points=grid_t, z_n=z, t_total=total_time, delta_t=0.8e-6)
    
    # --- Start timer ---
    start_time = time.time()
    
    surface, trajectory_points = sim.run()

    # --- End timer ---
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Simulation executed in {elapsed_time:.3f} seconds")
    
    print('Simulation End')


    plot_cutting_edge_trajectories(trajectory_points, plot_3d=False) # 2D
    plot_surface_scatter(surface, m=grid_m, n=grid_n)
    plot_cross_sections(surface, m=grid_m, n=grid_n)


    end = time.perf_counter()
    print(f"Total runtime: {end - start:.6f} seconds")
