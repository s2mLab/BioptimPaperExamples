from time import time

import biorbd
from bioptim import Solver

from .gait.load_experimental_data import LoadData
from .gait.ocp import prepare_ocp, get_phase_time_shooting_numbers, get_experimental_data


def generate_table(out):
    root_path = "/".join(__file__.split("/")[:-1])

    # Define the problem -- model path
    biorbd_model = (
        biorbd.Model(root_path + "/models/Gait_1leg_12dof_heel.bioMod"),
        biorbd.Model(root_path + "/models/Gait_1leg_12dof_flatfoot.bioMod"),
        biorbd.Model(root_path + "/models/Gait_1leg_12dof_forefoot.bioMod"),
        biorbd.Model(root_path + "/models/Gait_1leg_12dof_0contact.bioMod"),
    )

    # --- files path ---
    c3d_file = root_path + "/data/normal01_out.c3d"
    q_kalman_filter_file = root_path + "/data/normal01_q_KalmanFilter.txt"
    qdot_kalman_filter_file = root_path + "/data/normal01_qdot_KalmanFilter.txt"
    data = LoadData(biorbd_model[0], c3d_file, q_kalman_filter_file, qdot_kalman_filter_file)

    # --- phase time and number of shooting ---
    phase_time, number_shooting_points = get_phase_time_shooting_numbers(data, 0.01)
    # --- get experimental data ---
    q_ref, qdot_ref, markers_ref, grf_ref, moments_ref, cop_ref = get_experimental_data(data, number_shooting_points)

    ocp = prepare_ocp(
        biorbd_model=biorbd_model,
        final_time=phase_time,
        nb_shooting=number_shooting_points,
        markers_ref=markers_ref,
        grf_ref=grf_ref,
        q_ref=q_ref,
        qdot_ref=qdot_ref,
        M_ref=moments_ref,
        CoP=cop_ref,
        nb_threads=8,
    )

    # --- Solve the program --- #
    tic = time()
    sol = ocp.solve(
        solver=Solver.IPOPT,
        solver_options={
            "tol": 1e-3,
            "max_iter": 1000,
            "hessian_approximation": "exact",
            "limited_memory_max_history": 50,
            "linear_solver": "ma57",
        },
    )
    toc = time() - tic
    sol_merged = sol.merge_phases()

    out.nx = sol_merged.states["all"].shape[0]
    out.nu = sol_merged.controls["all"].shape[0]
    out.ns = sol_merged.ns[0]
    out.solver.append(out.Solver("Ipopt"))
    out.solver[0].n_iteration = sol.iterations
    out.solver[0].cost = sol.cost
    out.solver[0].convergence_time = toc
    out.solver[0].compute_error_single_shooting(sol, 1, use_final_time=True)
