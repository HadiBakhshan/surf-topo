import json
import numpy as np
import h5py
import surftopo
import sys
import os 
import time

# ==========================================================
# LOAD CONFIG
# ==========================================================

def load_config(path):
    with open(path, "r") as f:
        config = json.load(f)   # dictionary containing simulation settings
    return config


# ==========================================================
# LHS SAMPLER
# ==========================================================

def lhs(n_samples, n_dim, seed=None):
    rng = np.random.default_rng(seed)          # random number generator (reproducible if seed given)
    result = np.zeros((n_samples, n_dim))      # LHS matrix to hold samples

    for i in range(n_dim):
        perm = rng.permutation(n_samples)     # stratified permutation for LHS
        result[:, i] = (perm + rng.random(n_samples)) / n_samples  # normalize to [0,1]

    return result


# ==========================================================
# SAMPLE GENERATION
# ==========================================================

def generate_samples(config):
    input_cfg = config["input"]                # dictionary of input ranges
    sampling_cfg = config["sampling"]          # sampling configuration

    n_samples = sampling_cfg["n_samples"]      # number of samples to generate
    seed = sampling_cfg.get("seed", None)      # random seed

    continuous_keys = []                        # keys of continuous variables
    discrete_keys = []                          # keys of discrete variables

    # classify input variables as continuous or discrete
    for key, values in input_cfg.items():
        if len(values) == 2 and isinstance(values[0], float):
            continuous_keys.append(key)        # continuous: range [min, max]
        else:
            discrete_keys.append(key)          # discrete: select from list

    lhs_unit = lhs(n_samples, len(continuous_keys), seed)  # unit LHS values in [0,1]
    rng = np.random.default_rng(seed)

    samples = []

    for i in range(n_samples):
        sample = {}

        # assign continuous variables
        for j, key in enumerate(continuous_keys):
            vmin, vmax = input_cfg[key]
            sample[key] = float(vmin + lhs_unit[i, j] * (vmax - vmin))  # linear interpolation

        # assign discrete variables
        for key in discrete_keys:
            sample[key] = rng.choice(input_cfg[key])  # randomly select discrete value

        samples.append(sample)

    return samples


# ==========================================================
# DERIVED PARAMETERS
# ==========================================================

def compute_derived(sample, grid_cfg, terminal_output = False):
    vc = sample["vc"]              # cutting speed (m/min)
    Td = sample["Td"]              # tool diameter (mm)
    fz = sample["fz"]              # feed per tooth (mm/tooth)
    z = sample["z"]                # number of teeth / cutting edges

    ws = vc / (Td * 0.001 * np.pi) # spindle speed (RPM)
    vf = fz * z * ws / 60          # feed velocity (mm/s)
    omega = ws * np.pi / 30        # angular velocity (rad/s)

    grid_base = grid_cfg["grid_base"] # base grid resolution
    length = grid_cfg["length"]       # workpiece length (mm)

    grid_m = grid_base * 2           # number of grid points along X
    grid_n = grid_base               # number of grid points along Y
    Lx = length * 2                  # workpiece total length in X
    Ly = length                       # workpiece total length in Y
    grid_t = max(grid_m, grid_n) * 2  # tool edge discretization points
    total_time = (1.5 * Td + Ly) / vf # estimated total cutting time (s)

    if terminal_output:
        print('Spindle velocity:', round(ws,2), 'RPM')
        print('Feed speed:', round(vf,2), 'mm/s')
        print('Milling time:', round(total_time, 2), 's')

    return {
        "omega": omega,               # spindle angular speed (rad/s)
        "v_f": vf,                    # feed velocity (mm/s)
        "grid_m": grid_m,             # number of X grid points
        "grid_n": grid_n,             # number of Y grid points
        "Lx": Lx,                     # workpiece X length (mm)
        "Ly": Ly,                     # workpiece Y length (mm)
        "grid_t": grid_t,             # tool edge discretization
        "t_total": total_time         # total cutting time (s)
    }

def create_tool_from_sample(sample, config):
    """Create a Tool instance from sample and configuration."""
    terminal_output = True
    derived = compute_derived(sample, config["grid"], terminal_output)
    fixed = config["fixed"]

    tool = surftopo.Tool(
        gama_f=sample["gama_f"],
        gama_p=sample["gama_p"],
        D=sample["Td"],
        eps_r=fixed["eps_r"],
        eps_a=fixed["eps_a"],
        phi=fixed["phi"],
        omega=derived["omega"],
        v_f=derived["v_f"],
        x_0=fixed["x_0"],
        y_0=fixed["y_0"],
        z_0=fixed["z_0"],
        fz=sample["fz"],
        ri=sample["ri"],
        z_n=sample["z"]
    )

    return tool, derived

# ==========================================================
# RUN ONE SIMULATION
# ==========================================================

def run_surface_simulation(sample, config):
    """Compute only surface points."""
    tool, derived = create_tool_from_sample(sample, config)

    return surftopo.run_surface_simulation(
        tool,
        d_x=derived["Lx"], m=derived["grid_m"],
        d_y=derived["Ly"], n=derived["grid_n"],
        z_level=sample["ap"],
        edge_points=derived["grid_t"],
        t_total=derived["t_total"],
        delta_t=config["integration"]["delta_t"]
    )


def run_trajectory_simulation(sample, config, trajectory_downsample=1):
    """Compute only trajectory points."""
    tool, derived = create_tool_from_sample(sample, config)

    return surftopo.run_trajectory_simulation(
        tool,
        d_x=derived["Lx"], m=derived["grid_m"],
        d_y=derived["Ly"], n=derived["grid_n"],
        z_level=sample["ap"],
        edge_points=derived["grid_t"],
        t_total=derived["t_total"],
        delta_t=config["integration"]["delta_t"],
        trajectory_downsample=trajectory_downsample
    )


def run_simulation_case(sample, config, trajectory_downsample=1):
    """Run both surface and trajectory simulations and return them together."""
    surface = run_surface_simulation(sample, config)
    trajectory = run_trajectory_simulation(sample, config, trajectory_downsample) \
        if trajectory_downsample > 0 else None
    return surface, trajectory


# ==========================================================
# MAIN EXECUTION
# ==========================================================

def main(config_path):
    config = load_config(config_path)

    # --- Create folder for the case ---
    case_name = config.get("case_name", "default_case")  # folder name for this simulation
    folder = os.path.join(os.getcwd(), case_name)        # full path
    os.makedirs(folder, exist_ok=True)
    print(f"Output folder: {folder}")

    # --- Single case mode ---
    if config["mode"] == "single":
        from scripts.plot_utils import (
            plot_cutting_edge_trajectories,
            plot_surface_scatter,
            plot_cross_sections
        )

        sample = config["input"]              # input parameters for single simulation
 
        # --- Start timer ---
        start_time = time.time()

        # Run the simulation
        surface = run_surface_simulation(sample, config)

        # --- End timer ---
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Surface simulation executed in {elapsed_time:.3f} seconds")        
            
        print("Surface shape:", surface.shape)        

        # --- Cutting edge trajectories ---
        trajectory_downsample = 1

        start_time = time.time()
        trajectory = run_trajectory_simulation(sample, config, trajectory_downsample)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Trajectory simulation executed in {elapsed_time:.3f} seconds") 

        if len(trajectory)>0:            
            print("Trajectory points:", len(trajectory))
            plot_cutting_edge_trajectories(trajectory, plot_3d=False,
                                        save_path=os.path.join(folder, "cutting_edge_trajectories_2D.png"))

            plot_cutting_edge_trajectories(trajectory, plot_3d=True,
                                        save_path=os.path.join(folder, "cutting_edge_trajectories_3D.png"))

        
        m = compute_derived(sample, config["grid"])["grid_m"]  # X grid points
        n = compute_derived(sample, config["grid"])["grid_n"]  # Y grid points

        # --- Surface scatter ---
        plot_surface_scatter(surface, m, n,
                            save_path=os.path.join(folder, "surface_scatter.png"))

        # --- Cross-sections ---
        plot_cross_sections(surface, m, n,
                            save_path=os.path.join(folder, "cross_sections.png"))

        print(f"All plots saved in folder: {folder}")
        return

    # --- Dataset mode ---
    elif config["mode"] == "dataset":
        samples = generate_samples(config)

        first_surface = run_surface_simulation(samples[0], config)
        H, W = first_surface.shape              # surface array dimensions

        n_samples = len(samples)                # total number of simulations
        n_inputs = len(samples[0])              # number of input variables

        output_path = os.path.join(folder, "dataset.h5")  # save dataset inside folder

        with h5py.File(output_path, "w") as f:
            X_ds = f.create_dataset("X", (n_samples, n_inputs), dtype=np.float32)
            Y_ds = f.create_dataset("Y", (n_samples, H, W), dtype=np.float32)

            keys = list(samples[0].keys())
            f.attrs["input_keys"] = json.dumps(keys)

            for i, sample in enumerate(samples):
                print(f"Running {i+1}/{n_samples}")
                surface = run_surface_simulation(sample, config)
                X_ds[i] = np.array([sample[k] for k in keys], dtype=np.float32)  # input array
                Y_ds[i] = surface.astype(np.float32)                               # output surface

        print("Dataset saved to:", output_path)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python dataset_builder.py config_test.json")
        print("  python dataset_builder.py config_dataset.json")
        sys.exit(1)

    config_path = sys.argv[1]
    main(config_path)
