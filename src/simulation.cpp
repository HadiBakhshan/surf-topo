#include "simulation.hpp"
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <atomic>
#include <vector>
#include <limits>

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

//
// =========================
// Tool Implementation
// =========================
//

Tool::Tool(double gama_f_, double gama_p_, double D_,
           double eps_r_, double eps_a_, double phi_,
           double omega_, double v_f_,
           double x0, double y0, double z0,
           double fz_, double ri_,
           int z_n_)
    : gama_f(gama_f_), gama_p(gama_p_), D(D_),
      eps_r(eps_r_), eps_a(eps_a_), phi(phi_),
      omega(omega_), v_f(v_f_),
      x_0(x0), y_0(y0), z_0(z0),
      fz(fz_), ri(ri_), z_n(z_n_)
{
    constexpr double deg_to_rad = M_PI / 180.0;

    gamma_f_rad = gama_f * deg_to_rad;
    gamma_p_rad = gama_p * deg_to_rad;
    phi_rad     = phi    * deg_to_rad;

    cos_gamma_p = std::cos(gamma_p_rad);
    sin_gamma_p = std::sin(gamma_p_rad);

    // -----------------------------
    // Precompute delta offsets for each cutting edge
    // -----------------------------
    delta_r.resize(z_n);
    delta_a.resize(z_n);

    for (int K = 0; K < z_n; ++K)
    {
        delta_r[K] = D / 2.0 + K * eps_r;  // <-- here
        delta_a[K] = K * eps_a;
    }
}

Eigen::Matrix4d Tool::build_final_transformation_matrix(int K,                                                        
                                                        double t) const
{
    // Use precomputed delta offsets (0-based indexing)
    const double dr = delta_r[K - 1];  // K assumed 1-based
    const double da = delta_a[K - 1];

    // Precompute rotation difference
    double var1 = phi_rad + 2.0 * M_PI * (K - 1) / z_n - omega * t;
    double cos_diff = std::cos(gamma_f_rad - var1);
    double sin_diff = std::sin(gamma_f_rad - var1);

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();

    T(0,0) = cos_diff;
    T(0,1) = cos_diff * cos_gamma_p;
    T(0,2) = cos_diff * sin_gamma_p;
    T(0,3) = cos_diff * dr + sin_diff * da * sin_gamma_p + x_0;

    T(1,0) = sin_diff;
    T(1,1) = sin_diff * cos_gamma_p;
    T(1,2) = sin_diff * sin_gamma_p;
    T(1,3) = sin_diff * dr + cos_diff * da * sin_gamma_p + y_0 + v_f * t;

    T(2,0) = 0.0;
    T(2,1) = -sin_gamma_p;
    T(2,2) =  cos_gamma_p;
    T(2,3) =  z_0 + da * cos_gamma_p;

    return T;
}

//
// =========================
// Simulation Implementation
// =========================
//

Simulation::Simulation(const Tool& tool_,
                       double dx, int m_, double dy, int n_, double z_,
                       int edge_pts, double t_total_, double delta_t_)
    : tool(tool_),
      d_x(dx), m(m_),
      d_y(dy), n(n_),
      z_level(z_),
      edge_points(edge_pts),
      t_total(t_total_),
      delta_t(delta_t_)
{
    const double R = tool.ri;
    const double gamma_f = tool.gamma_f_rad;
    const double fz = tool.fz;
    
    // --------------------------------------------------
    // Compute chord length L
    // --------------------------------------------------
    // Compute the half-chord length L from axial immersion: axial depth of cut z_level
    const double Lap = std::sqrt(std::max(0.0, R*R - (R - z_level)*(R - z_level)));
    
    // Approximate chord length required by feed per tooth
    const double Lfz = fz / (2.0 * std::cos(gamma_f));

    const double L = std::max(Lap, Lfz);    
    
    // Precompute tool edge points along X
    tool_points.resize(edge_points);
    for (int i = 0; i < edge_points; ++i)
    {
        double l = -L + 2.0 * L * i / (edge_points - 1);
        double z = R - std::sqrt(std::max(0.0, R*R - l*l));
        tool_points[i] = Eigen::Vector4d(l, 0.0, z, 1.0);
    }

}

// -----------------------------
// Compute surface points (parallelized)
// -----------------------------
void Simulation::compute_surface(std::vector<Eigen::Vector3d>& surface_points) const
{
    const int grid_size = (m + 1) * (n + 1);
    const double delta_x = d_x / m;
    const double delta_y = d_y / n;

    // Atomic Z grid
    std::vector<std::atomic<double>> Z(grid_size);
    for (auto& v : Z) v.store(z_level, std::memory_order_relaxed);

    const int num_time_steps = static_cast<int>(t_total / delta_t) + 1;
    const int z_n = tool.z_n;

    // -----------------------------
    // OpenMP parallel loop over time steps
    // -----------------------------
    #pragma omp parallel
    {
        std::vector<double> local_min(grid_size, std::numeric_limits<double>::max());
        std::vector<int> touched_cells;
        touched_cells.reserve(edge_points * z_n);

        #pragma omp for schedule(dynamic)
        for (int t_idx = 0; t_idx < num_time_steps; ++t_idx)
        {
            double t = t_idx * delta_t;

            for (int K = 1; K <= z_n; ++K)
            {
                Eigen::Matrix4d T = tool.build_final_transformation_matrix(K, t);

                for (int p = 0; p < edge_points; ++p)
                {
                    Eigen::Vector4d tp = T * tool_points[p];
                    double xp = tp(0), yp = tp(1), zp = tp(2);

                    int xi = std::clamp(int(xp / delta_x), 0, m);
                    int yi = std::clamp(int(yp / delta_y), 0, n);
                    int idx = xi * (n + 1) + yi;

                    if (local_min[idx] == std::numeric_limits<double>::max())
                        touched_cells.push_back(idx);

                    if (zp < local_min[idx])
                        local_min[idx] = zp;
                }
            }

            // Atomic update of Z grid per touched cell
            for (int idx : touched_cells)
            {
                double zp = local_min[idx];
                double old_val = Z[idx].load(std::memory_order_relaxed);
                while (zp < old_val &&
                       !Z[idx].compare_exchange_weak(old_val, zp, std::memory_order_relaxed))
                {}
                local_min[idx] = std::numeric_limits<double>::max();
            }

            touched_cells.clear();
        }
    }

    // -----------------------------
    // Convert atomic Z grid to surface points (parallel)
    // -----------------------------
    surface_points.resize(grid_size);
    #pragma omp parallel for
    for (int i = 0; i <= m; ++i)
    {
        for (int j = 0; j <= n; ++j)
        {
            int idx = i * (n + 1) + j;
            surface_points[idx] = Eigen::Vector3d(i * delta_x, j * delta_y,
                                                 Z[idx].load(std::memory_order_relaxed));
        }
    }
}

// -----------------------------
// Compute trajectory points (parallel, order preserved)
// -----------------------------
void Simulation::compute_trajectory(std::vector<TrajectoryPoint>& trajectory_points,
                                     int trajectory_downsample) const
{
    if (trajectory_downsample <= 0) return;

    const int num_time_steps = static_cast<int>(t_total / delta_t) + 1;
    const int z_n = tool.z_n;
    const int num_output_steps = (num_time_steps + trajectory_downsample - 1) / trajectory_downsample;

    trajectory_points.resize(num_output_steps * z_n); // preallocate

    #pragma omp parallel for schedule(static)
    for (int t_out_idx = 0; t_out_idx < num_output_steps; ++t_out_idx)
    {
        int t_idx = t_out_idx * trajectory_downsample;
        double t = t_idx * delta_t;

        for (int K = 1; K <= z_n; ++K)
        {
            // Compute tool transform
            Eigen::Matrix4d T = tool.build_final_transformation_matrix(K, t);

            // Find minimal Z point along the edge
            double min_z = std::numeric_limits<double>::max();
            Eigen::Vector3d min_point;

            for (int p = 0; p < edge_points; ++p)
            {
                Eigen::Vector4d tp = T * tool_points[p];
                double zp = tp(2);
                if (zp < min_z)
                {
                    min_z = zp;
                    min_point = Eigen::Vector3d(tp(0), tp(1), zp);
                }
            }

            // Insert into the preallocated vector at the correct index
            int global_idx = t_out_idx * z_n + (K - 1);
            trajectory_points[global_idx] = {K, t, min_point(0), min_point(1), min_point(2)};
        }
    }
}

// -----------------------------
// Original run method now simply calls the two methods
// -----------------------------
void Simulation::run(std::vector<Eigen::Vector3d>& surface_points,
                     std::vector<TrajectoryPoint>& trajectory_points,
                     int trajectory_downsample) const
{
    compute_surface(surface_points);
    compute_trajectory(trajectory_points, trajectory_downsample);
}
