#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "simulation.hpp"

namespace py = pybind11;

// Full simulation: surface + trajectory
py::tuple run_simulation(
    const Tool &tool,
    double d_x, int m,
    double d_y, int n,
    double z_level,
    int edge_points,    
    double t_total, 
    double delta_t,
    int trajectory_downsample)
{
    Simulation sim(tool, d_x, m, d_y, n, z_level,
                   edge_points, t_total, delta_t);

    std::vector<Eigen::Vector3d> surface_points;
    std::vector<TrajectoryPoint> trajectory_points;

    sim.run(surface_points, trajectory_points, trajectory_downsample);

    // --- Convert surface_points to NumPy array ---
    std::vector<py::ssize_t> shape_surface = {
        static_cast<py::ssize_t>(surface_points.size()), 3
    };
    py::array_t<double> np_surface(shape_surface);
    auto buf_surface = np_surface.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(surface_points.size()); ++i) {
        buf_surface(i, 0) = surface_points[i][0];
        buf_surface(i, 1) = surface_points[i][1];
        buf_surface(i, 2) = surface_points[i][2];
    }

    // --- Convert trajectory_points to NumPy array ---
    std::vector<py::ssize_t> shape_traj = {
        static_cast<py::ssize_t>(trajectory_points.size()), 5
    };
    py::array_t<double> np_traj(shape_traj);
    auto buf_traj = np_traj.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(trajectory_points.size()); ++i) {
        buf_traj(i, 0) = trajectory_points[i].K;
        buf_traj(i, 1) = trajectory_points[i].t;
        buf_traj(i, 2) = trajectory_points[i].x;
        buf_traj(i, 3) = trajectory_points[i].y;
        buf_traj(i, 4) = trajectory_points[i].z;
    }

    return py::make_tuple(np_surface, np_traj);
}

// Surface only
py::array_t<double> run_surface_simulation(
    const Tool &tool,
    double d_x, int m,
    double d_y, int n,
    double z_level,
    int edge_points,    
    double t_total, 
    double delta_t)
{
    Simulation sim(tool, d_x, m, d_y, n, z_level,
                   edge_points, t_total, delta_t);

    std::vector<Eigen::Vector3d> surface_points;
    sim.compute_surface(surface_points);

    std::vector<py::ssize_t> shape_surface = {
        static_cast<py::ssize_t>(surface_points.size()), 3
    };
    py::array_t<double> np_surface(shape_surface);
    auto buf_surface = np_surface.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(surface_points.size()); ++i) {
        buf_surface(i, 0) = surface_points[i][0];
        buf_surface(i, 1) = surface_points[i][1];
        buf_surface(i, 2) = surface_points[i][2];
    }

    return np_surface;
}

// Trajectory only
py::array_t<double> run_trajectory_simulation(
    const Tool &tool,
    double d_x, int m,
    double d_y, int n,
    double z_level,
    int edge_points,    
    double t_total, 
    double delta_t,
    int trajectory_downsample = 1)
{
    Simulation sim(tool, d_x, m, d_y, n, z_level,
                   edge_points, t_total, delta_t);

    std::vector<TrajectoryPoint> trajectory_points;
    sim.compute_trajectory(trajectory_points, trajectory_downsample);

    std::vector<py::ssize_t> shape_traj = {
        static_cast<py::ssize_t>(trajectory_points.size()), 5
    };
    py::array_t<double> np_traj(shape_traj);
    auto buf_traj = np_traj.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(trajectory_points.size()); ++i) {
        buf_traj(i, 0) = trajectory_points[i].K;
        buf_traj(i, 1) = trajectory_points[i].t;
        buf_traj(i, 2) = trajectory_points[i].x;
        buf_traj(i, 3) = trajectory_points[i].y;
        buf_traj(i, 4) = trajectory_points[i].z;
    }

    return np_traj;
}

// -----------------------------
// Module definition
// -----------------------------
PYBIND11_MODULE(machtopo, m) {
    m.doc() = "High-performance milling simulation";

    py::class_<Tool>(m, "Tool")
        .def(py::init<
            double,double,double,double,double,double,
            double,double,double,double,double,double,double,int>(),
            py::arg("gama_f"),
            py::arg("gama_p"),
            py::arg("D"),
            py::arg("eps_r"),
            py::arg("eps_a"),
            py::arg("phi"),
            py::arg("omega"),
            py::arg("v_f"),
            py::arg("x_0"),
            py::arg("y_0"),
            py::arg("z_0"),
            py::arg("fz"),
            py::arg("ri"),
            py::arg("z_n")
        );

    py::class_<Simulation>(m, "Simulation")
        .def(py::init<const Tool&, double,int,double,int,double,int,double,double>(),
             py::arg("tool"),
             py::arg("d_x"), py::arg("m"),
             py::arg("d_y"), py::arg("n"),
             py::arg("z_level"),
             py::arg("edge_points"),
             py::arg("t_total"),
             py::arg("delta_t"))
        .def("run", &Simulation::run,
             py::arg("surface_points"),
             py::arg("trajectory_points"),
             py::arg("trajectory_downsample") = 1)
        .def("compute_surface", &Simulation::compute_surface,
             py::arg("surface_points"))
        .def("compute_trajectory", &Simulation::compute_trajectory,
             py::arg("trajectory_points"),
             py::arg("trajectory_downsample") = 1);

    m.def("run_simulation", &run_simulation,
          "Run full simulation (surface + trajectory)",
          py::arg("tool"),
          py::arg("d_x"), py::arg("m"),
          py::arg("d_y"), py::arg("n"),
          py::arg("z_level"),
          py::arg("edge_points"),          
          py::arg("t_total"), 
          py::arg("delta_t"),
          py::arg("trajectory_downsample"));

    m.def("run_surface_simulation", &run_surface_simulation,
          "Run only surface simulation",
          py::arg("tool"),
          py::arg("d_x"), py::arg("m"),
          py::arg("d_y"), py::arg("n"),
          py::arg("z_level"),
          py::arg("edge_points"),          
          py::arg("t_total"), 
          py::arg("delta_t"));

    m.def("run_trajectory_simulation", &run_trajectory_simulation,
          "Run only trajectory simulation",
          py::arg("tool"),
          py::arg("d_x"), py::arg("m"),
          py::arg("d_y"), py::arg("n"),
          py::arg("z_level"),
          py::arg("edge_points"),          
          py::arg("t_total"), 
          py::arg("delta_t"),
          py::arg("trajectory_downsample") = 1);
}

/* 
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "simulation.hpp"

namespace py = pybind11;

py::tuple run_simulation(
    const Tool &tool,
    double d_x, int m,
    double d_y, int n,
    double z_level,
    int edge_points,    
    double t_total, 
    double delta_t,
    int trajectory_downsample)
{
    Simulation sim(tool, d_x, m, d_y, n, z_level,
                   edge_points, t_total, delta_t);

    std::vector<Eigen::Vector3d> surface_points;
    std::vector<TrajectoryPoint> trajectory_points;

    sim.run(surface_points, trajectory_points, trajectory_downsample);

    // --- Convert surface_points to NumPy array ---
    std::vector<py::ssize_t> shape_surface = {
        static_cast<py::ssize_t>(surface_points.size()), 3
    };
    py::array_t<double> np_surface(shape_surface);
    auto buf_surface = np_surface.mutable_unchecked<2>();

    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(surface_points.size()); ++i) {
        buf_surface(i, 0) = surface_points[i][0]; // X
        buf_surface(i, 1) = surface_points[i][1]; // Y
        buf_surface(i, 2) = surface_points[i][2]; // Z
    }
    
    // --- Convert trajectory_points to NumPy array ---
    std::vector<ssize_t> shape_traj = { static_cast<ssize_t>(trajectory_points.size()), 5 };
    py::array_t<double> np_traj(shape_traj);
    auto buf_traj = np_traj.mutable_unchecked<2>();
    for (ssize_t i = 0; i < static_cast<ssize_t>(trajectory_points.size()); ++i) {
        buf_traj(i, 0) = trajectory_points[i].K;
        buf_traj(i, 1) = trajectory_points[i].t;
        buf_traj(i, 2) = trajectory_points[i].x;
        buf_traj(i, 3) = trajectory_points[i].y;
        buf_traj(i, 4) = trajectory_points[i].z;
    }


    return py::make_tuple(np_surface, np_traj);
}

PYBIND11_MODULE(machtopo, m) {
    m.doc() = "High-performance milling simulation";

    py::class_<Tool>(m, "Tool")
    .def(py::init<
        double,double,double,double,double,double,
        double,double,double,double,double,double,double, int>(),
        py::arg("gama_f"),
        py::arg("gama_p"),
        py::arg("D"),
        py::arg("eps_r"),
        py::arg("eps_a"),
        py::arg("phi"),
        py::arg("omega"),
        py::arg("v_f"),
        py::arg("x_0"),
        py::arg("y_0"),
        py::arg("z_0"),
        py::arg("fz"),
        py::arg("ri"),
        py::arg("z_n")
    );
    //.def("build_final_transformation_matrix", &Tool::build_final_transformation_matrix);

    m.def("run_simulation", &run_simulation,
          "Run the simulation",
          py::arg("tool"),
          py::arg("d_x"), py::arg("m"),
          py::arg("d_y"), py::arg("n"),
          py::arg("z_level"),
          py::arg("edge_points"),          
          py::arg("t_total"), 
          py::arg("delta_t"),
          py::arg("trajectory_down_sample"));
}
 */