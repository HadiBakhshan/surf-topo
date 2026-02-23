#pragma once
#include <Eigen/Dense>
#include <vector>

// -----------------------------
// Tool class
// -----------------------------

#pragma once
#include <vector>
#include <Eigen/Dense>

class Tool {
public:
    // Constructor
    Tool(double gama_f_, double gama_p_, double D_,
         double eps_r_, double eps_a_, double phi_, double omega_, double v_f_,
         double x0, double y0, double z0, double fz_, double ri_,
         int z_n_);

    // Optional: Eigen transformation (legacy/debug)
    Eigen::Matrix4d build_final_transformation_matrix(int K, double t) const;

    // Tool parameters
    double gama_f, gama_p, D;
    double eps_r, eps_a, phi, omega, v_f;
    double x_0, y_0, z_0, fz, ri;
    int z_n;

    // Precomputed cutting-edge offsets
    std::vector<double> delta_r;  // radial offset per cutting edge
    std::vector<double> delta_a;  // axial offset per cutting edge

    // Cached radian and trig values
    double gamma_f_rad;
    double gamma_p_rad;
    double phi_rad;
    double cos_gamma_p;
    double sin_gamma_p;
};


// -----------------------------
// Trajectory point struct
// -----------------------------
struct TrajectoryPoint {
    int K;
    double t, x, y, z;
};

class Simulation {
public:
    const Tool& tool;
    double d_x, d_y;
    int m, n;
    double z_level;    
    int edge_points;
    double t_total, delta_t;

    // Precomputed tool edge points
    std::vector<Eigen::Vector4d> tool_points;

    Simulation(const Tool& tool_,
               double dx, int m_, double dy, int n_, double z_,
               int edge_pts, double t_total_, double delta_t_);

    void run(std::vector<Eigen::Vector3d>& surface_points,
             std::vector<TrajectoryPoint>& trajectory_points,
             int trajectory_downsample = 1) const;

    void compute_surface(std::vector<Eigen::Vector3d>& surface_points) const;
    void compute_trajectory(std::vector<TrajectoryPoint>& trajectory_points,
                            int trajectory_downsample) const;
};

