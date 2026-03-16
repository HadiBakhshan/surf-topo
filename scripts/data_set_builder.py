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
        config = json.load(f)
    return config


# ==========================================================
# LHS SAMPLER
# ==========================================================

def lhs(n_samples, n_dim, seed=None):
    rng = np.random.default_rng(seed)
    result = np.zeros((n_samples, n_dim))

    for i in range(n_dim):
        perm = rng.permutation(n_samples)
        result[:, i] = (perm + rng.random(n_samples)) / n_samples

    return result


# ==========================================================
# SAMPLE GENERATION
# ==========================================================

def generate_samples(config):
    input_cfg = config["input"]
    sampling_cfg = config["sampling"]

    n_samples = sampling_cfg["n_samples"]
    seed = sampling_cfg.get("seed", None)

    continuous_keys = []
    discrete_keys = []

    for key, values in input_cfg.items():
        if len(values) == 2 and isinstance(values[0], float):
            continuous_keys.append(key)
        else:
            discrete_keys.append(key)

    lhs_unit = lhs(n_samples, len(continuous_keys), seed)
    rng = np.random.default_rng(seed)

    samples = []

    for i in range(n_samples):
        sample = {}

        # continuous variables
        for j, key in enumerate(continuous_keys):
            vmin, vmax = input_cfg[key]
            sample[key] = float(vmin + lhs_unit[i, j] * (vmax - vmin))

        # discrete variables
        for key in discrete_keys:
            sample[key] = rng.choice(input_cfg[key])

        samples.append(sample)

    return samples


# ==========================================================
# DERIVED PARAMETERS
# ==========================================================

def compute_derived(sample, config, grid_cfg, terminal_output=False):

    fixed = config["fixed"]

    vc = sample["vc"]
    Td = fixed["Td"]
    fz = sample["fz"]
    z = sample["z"]

    ws = vc / (Td * 0.001 * np.pi)
    vf = fz * z * ws / 60
    omega = ws * np.pi / 30

    grid_base = grid_cfg["grid_base"]
    length = grid_cfg["length"]

    grid_m = grid_base * 2
    grid_n = grid_base

    Lx = length * 1.31
    Ly = length

    grid_t = max(grid_m, grid_n) * 2
    total_time = (1.5 * Td + Ly) / vf

    if terminal_output:
        print('Spindle velocity:', round(ws, 2), 'RPM')
        print('Feed speed:', round(vf, 2), 'mm/s')
        print('Milling time:', round(total_time, 2), 's')

    return {
        "omega": omega,
        "v_f": vf,
        "grid_m": grid_m,
        "grid_n": grid_n,
        "Lx": Lx,
        "Ly": Ly,
        "grid_t": grid_t,
        "t_total": total_time
    }


# ==========================================================
# TOOL CREATION
# ==========================================================

def create_tool_from_sample(sample, config):

    derived = compute_derived(sample, config, config["grid"])
    fixed = config["fixed"]

    tool = surftopo.Tool(
        gama_f=fixed["gama_f"],
        gama_p=fixed["gama_p"],
        D=fixed["Td"],
        eps_r=sample["eps_r"],
        eps_a=sample["eps_a"],
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

    tool, derived = create_tool_from_sample(sample, config)

    return surftopo.run_surface_simulation(
        tool,
        d_x=derived["Lx"], m=derived["grid_m"],
        d_y=derived["Ly"], n=derived["grid_n"],
        z_level=config["fixed"]["ap"],
        edge_points=derived["grid_t"],
        t_total=derived["t_total"],
        delta_t=config["integration"]["delta_t"]
    )


def run_trajectory_simulation(sample, config, trajectory_downsample=1):

    tool, derived = create_tool_from_sample(sample, config)

    return surftopo.run_trajectory_simulation(
        tool,
        d_x=derived["Lx"], m=derived["grid_m"],
        d_y=derived["Ly"], n=derived["grid_n"],
        z_level=config["fixed"]["ap"],
        edge_points=derived["grid_t"],
        t_total=derived["t_total"],
        delta_t=config["integration"]["delta_t"],
        trajectory_downsample=trajectory_downsample
    )


def run_simulation_case(sample, config, trajectory_downsample=1):

    surface = run_surface_simulation(sample, config)

    trajectory = None
    if trajectory_downsample > 0:
        trajectory = run_trajectory_simulation(sample, config, trajectory_downsample)

    return surface, trajectory


# ==========================================================
# MAIN EXECUTION
# ==========================================================

def main(config_path):

    config = load_config(config_path)

    case_name = config.get("case_name", "default_case")
    folder = os.path.join(os.getcwd(), case_name)

    os.makedirs(folder, exist_ok=True)
    print(f"Output folder: {folder}")


# ==========================================================
# SINGLE SIMULATION MODE
# ==========================================================

    if config["mode"] == "single":

        from scripts.plot_utils import (
            plot_cutting_edge_trajectories,
            plot_surface_scatter,
            plot_surface_topview,
            plot_cross_sections
        )

        sample = config["input"]

        start_time = time.time()
        surface = run_surface_simulation(sample, config)
        print("Surface simulation time:", time.time() - start_time)

        trajectory_downsample = 1

        start_time = time.time()
        trajectory = run_trajectory_simulation(sample, config, trajectory_downsample)
        print("Trajectory simulation time:", time.time() - start_time)

        m = compute_derived(sample, config, config["grid"])["grid_m"]
        n = compute_derived(sample, config, config["grid"])["grid_n"]

        plot_surface_scatter(surface, m, n,
            save_path=os.path.join(folder, "Surface_3D.png"))

        plot_surface_topview(surface, m, n,
            save_path=os.path.join(folder, "Surface_2D.png"))

        # plot_cross_sections(surface, m, n,
        #     save_path=os.path.join(folder, "Cross_Sections.png"))

        if trajectory is not None and len(trajectory) > 0:

            plot_cutting_edge_trajectories(
                trajectory,
                plot_3d=False,
                save_path=os.path.join(folder, "Trajectories_2D.png")
            )

            # plot_cutting_edge_trajectories(
            #     trajectory,
            #     plot_3d=True,
            #     save_path=os.path.join(folder, "Trajectories_3D.png")
            # )

        print(f"All plots saved in folder: {folder}")

        return


# ==========================================================
# DATASET MODE
# ==========================================================

    elif config["mode"] == "dataset":

        samples = generate_samples(config)

        n_samples = len(samples)
        n_inputs = len(samples[0])

        output_path = os.path.join(folder, "dataset_large.h5")

        with h5py.File(output_path, "w") as f:

            # -----------------------------
            # Create datasets
            # -----------------------------
            X_ds = f.create_dataset("X", (n_samples, n_inputs), dtype=np.float32)
            Y_ds = f.create_dataset("Y", (n_samples, 3), dtype=np.float32)

            keys = list(samples[0].keys())

            f.attrs["input_keys"] = json.dumps(keys)
            f.attrs["output_keys"] = json.dumps(["Ra", "Rq", "Rz"])

            # -----------------------------
            # Loop through samples
            # -----------------------------
            for i, sample in enumerate(samples):

                print(f"Running {i+1}/{n_samples}")

                # Run simulation
                surface = run_surface_simulation(sample, config)

                # -----------------------------
                # Extract Z values only and convert to µm
                # -----------------------------
                Z_mm = surface[:, 2]           # Only Z column
                Z_um = Z_mm * 1000             # Convert to µm
                z_flat = Z_um[~np.isnan(Z_um)] # Remove NaNs

                # -----------------------------
                # Compute roughness metrics
                # -----------------------------
                z_mean = np.mean(z_flat)

                Ra = np.mean(np.abs(z_flat - z_mean))
                Rq = np.sqrt(np.mean((z_flat - z_mean)**2))
                Rz = np.max(z_flat) - np.min(z_flat)

                # -----------------------------
                # Save inputs and outputs
                # -----------------------------
                X_ds[i] = np.array([sample[k] for k in keys], dtype=np.float32)
                Y_ds[i] = np.array([Ra, Rq, Rz], dtype=np.float32)

                # Optional: print first few samples for verification
                if i < 5:
                    print(f"Sample {i}: Ra={Ra:.4f}, Rq={Rq:.4f}, Rz={Rz:.4f}")

        print("Dataset saved to:", output_path)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage:")
        print("python dataset_builder.py config_test.json")
        print("python dataset_builder.py config_dataset.json")
        sys.exit(1)

    config_path = sys.argv[1]

    main(config_path)