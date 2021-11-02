"""
This is an example on gait biomechanics.
Experimental data (markers trajectories, ground reaction forces and moments) are tracked.
"""
from time import time

import numpy as np
import biorbd_casadi as biorbd
from bioptim import Solver, Shooting
import matplotlib.pyplot as plt

from gait.load_experimental_data import LoadData
from gait.ocp import prepare_ocp, get_phase_time_shooting_numbers, get_experimental_data


if __name__ == "__main__":
    root_path = "/".join(__file__.split("/")[:-1]) + "/"

    # Define the problem -- model path
    biorbd_model = (
        biorbd.Model(root_path + "models/Gait_1leg_12dof_heel.bioMod"),
        biorbd.Model(root_path + "models/Gait_1leg_12dof_flatfoot.bioMod"),
        biorbd.Model(root_path + "models/Gait_1leg_12dof_forefoot.bioMod"),
        biorbd.Model(root_path + "models/Gait_1leg_12dof_0contact.bioMod"),
    )

    # Problem parameters
    nb_q = biorbd_model[0].nbQ()
    nb_qdot = biorbd_model[0].nbQdot()
    nb_tau = biorbd_model[0].nbGeneralizedTorque()
    nb_phases = len(biorbd_model)
    nb_markers = biorbd_model[0].nbMarkers()

    # Generate data from file
    # --- files path ---
    c3d_file = root_path + "data/normal01_out.c3d"
    q_kalman_filter_file = root_path + "data/normal01_q_KalmanFilter.txt"
    qdot_kalman_filter_file = root_path + "data/normal01_qdot_KalmanFilter.txt"
    data = LoadData(biorbd_model[0], c3d_file, q_kalman_filter_file, qdot_kalman_filter_file)

    # --- phase time and number of shooting ---
    phase_time, number_shooting_points = get_phase_time_shooting_numbers(data, 0.01)
    # --- get experimental data ---
    q_ref, qdot_ref, markers_ref, grf_ref, moments_ref, cop_ref = get_experimental_data(data, number_shooting_points, phase_time)

    ocp = prepare_ocp(
        biorbd_model=biorbd_model,
        final_time=phase_time,
        nb_shooting=number_shooting_points,
        markers_ref=markers_ref,
        grf_ref=grf_ref,
        q_ref=q_ref,
        qdot_ref=qdot_ref,
        M_ref=moments_ref,
        nb_threads=8,
    )

    tic = time()
    # --- Solve the program --- #
    solver = Solver.IPOPT()
    solver.set_linear_solver("ma57")
    solver.set_tol(1e-3)
    solver.set_hessian_approximation("exact")
    solver.set_maximum_iterations(3000)
    solver.show_online_optim=True
    sol = ocp.solve(solver=solver)
    toc = time() - tic

    sol_ss = sol.integrate(shooting_type=Shooting.SINGLE_CONTINUOUS, merge_phases=False)
    ss_err_trans = np.sqrt(np.mean((sol_ss.states[-1]["q"][:3, -1] - sol.states[-1]["q"][:3, -1]) ** 2))
    ss_err_rot = np.sqrt(np.mean((sol_ss.states[-1]["q"][3:, -1] - sol.states[-1]["q"][3:, -1]) ** 2))

    print("*********************************************")
    # print(f"Problem solved with {solver.value}")
    print(f"Solving time : {toc} s")
    print(f"Single shooting error for translation: {ss_err_trans*1000} mm")
    print(f"Single shooting error for rotation: {ss_err_rot * 180/np.pi} degrees")
    ocp.save(sol, "gait.bo")

    # --- Show results --- #
    sol.animate(
        show_meshes=True,
        background_color=(1, 1, 1),
        show_local_ref_frame=False,
    )
    sol.graphs()


    # --- Save results --- #
    ocp.save(sol, "gait.bo")
