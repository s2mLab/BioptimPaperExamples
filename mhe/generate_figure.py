"""
This is a basic code to plot and animate the results of MHE example.

Please note that before to use this code you have to run main.py to generate results.
"""
from time import time
from math import ceil
import pickle

import seaborn
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import biorbd
import numpy as np
from bioptim import (
    BoundsList,
    QAndQDotBounds,
    InitialGuess,
    Solver,
    InterpolationType,
)

from mhe.ocp import force_func, generate_noise, prepare_ocp, define_objective, warm_start_mhe


def generate_data():
    """
    Prepare and solve the MHE example
    """
    use_noise = True  # True to add noise on reference joint angles
    root_path = "/".join(__file__.split("/")[:-1]) + "/"
    model = root_path + "/models/arm_wt_rot_scap.bioMod"
    t = 8
    ns = 800
    with open(f"{root_path}/data/sim_ac_8000ms_800sn_REACH2_co_level_0_step5_ERK.bob", "rb") as file:
        data = pickle.load(file)
    states = data["data"][0]
    controls = data["data"][1]
    q_ref = states["q"]
    dq_ref = states["qdot"]
    a_ref = states["muscles"]
    u_ref = controls["muscles"]

    biorbd_model = biorbd.Model(model)
    ns_mhe = 7
    rt_ratio = 3
    t_mhe = t / (ns / rt_ratio) * ns_mhe
    x_wt_noise = np.concatenate((q_ref, dq_ref))

    force_ref_tmp = np.ndarray((biorbd_model.nbMuscles(), u_ref.shape[1]))
    get_force = force_func(biorbd_model, use_activation=False)
    for i in range(biorbd_model.nbMuscles()):
        for k in range(u_ref.shape[1]):
            force_ref_tmp[i, k] = get_force(q_ref[:, k], dq_ref[:, k], a_ref[:, k], u_ref[:, k])[i, :]
    force_ref = force_ref_tmp[:, ::rt_ratio]

    q_noise = 5
    if use_noise:
        q_ref = generate_noise(biorbd_model, q_ref, q_noise)
    x_ref = np.concatenate((q_ref, dq_ref))

    x_est = np.zeros((biorbd_model.nbQ() * 2, x_ref[:, ::rt_ratio].shape[1] - ns_mhe))
    u_est = np.zeros((biorbd_model.nbMuscles(), u_ref[:, ::rt_ratio].shape[1] - ns_mhe))

    # Initial and final state
    get_force = force_func(biorbd_model)
    # --- Solve the program using ACADOS --- #
    ocp = prepare_ocp(biorbd_model=biorbd_model, final_time=t_mhe, n_shooting=ns_mhe)

    # Update bounds
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))
    x_bounds[0].min[: biorbd_model.nbQ(), 0] = x_ref[: biorbd_model.nbQ(), 0] - 0.1
    x_bounds[0].max[: biorbd_model.nbQ(), 0] = x_ref[: biorbd_model.nbQ(), 0] + 0.1
    ocp.update_bounds(x_bounds)

    # Update initial guess
    x_init = InitialGuess(x_ref[:, : ns_mhe + 1], interpolation=InterpolationType.EACH_FRAME)
    u_init = InitialGuess([0.2] * biorbd_model.nbMuscles(), interpolation=InterpolationType.CONSTANT)
    ocp.update_initial_guess(x_init, u_init)

    # Update objectives functions
    objectives = define_objective(q_ref, 0, rt_ratio, ns_mhe, biorbd_model)
    ocp.update_objectives(objectives)

    # Initialize the solver options
    sol = ocp.solve(
        solver=Solver.ACADOS,
        show_online_optim=False,
        solver_options={
            "nlp_solver_tol_comp": 1e-10,
            "nlp_solver_tol_eq": 1e-10,
            "nlp_solver_tol_stat": 1e-8,
            "integrator_type": "IRK",
            "nlp_solver_type": "SQP",
            "sim_method_num_steps": 1,
            "print_level": 0,
            "nlp_solver_max_iter": 30,
        },
    )

    # Set solutions and set initial guess for next optimisation
    x0, u0, x_est[:, 0], u_est[:, 0] = warm_start_mhe(sol)

    tic = time()  # Save initial time
    for i in range(1, x_est.shape[1]):
        # set initial state
        ocp.nlp[0].x_bounds.min[:, 0] = x0[:, 0]
        ocp.nlp[0].x_bounds.max[:, 0] = x0[:, 0]

        # Update initial guess
        x_init = InitialGuess(x0, interpolation=InterpolationType.EACH_FRAME)
        u_init = InitialGuess(u0, interpolation=InterpolationType.EACH_FRAME)
        ocp.update_initial_guess(x_init, u_init)

        # Update objectives functions
        objectives = define_objective(q_ref, i, rt_ratio, ns_mhe, biorbd_model)
        ocp.update_objectives(objectives)

        # Solve problem
        sol = ocp.solve(
            solver=Solver.ACADOS,
            show_online_optim=False,
            solver_options={
                "nlp_solver_tol_comp": 1e-6,
                "nlp_solver_tol_eq": 1e-6,
                "nlp_solver_tol_stat": 1e-5,
            },
        )
        # Set solutions and set initial guess for next optimisation
        x0, u0, x_out, u_out = warm_start_mhe(sol)
        x_est[:, i] = x_out
        if i < u_est.shape[1]:
            u_est[:, i] = u_out

    a_est = u_est
    q_ref = q_ref[:, ::rt_ratio]
    force_est = np.ndarray((biorbd_model.nbMuscles(), u_est.shape[1]))
    for i in range(biorbd_model.nbMuscles()):
        for k in range(u_est.shape[1]):
            force_est[i, k] = get_force(
                x_est[: biorbd_model.nbQ(), k],
                x_est[biorbd_model.nbQ() : biorbd_model.nbQ() * 2, k],
                a_est[:, k],
                u_est[:, k],
            )[i, :]

    toc = time() - tic
    final_offset = 5  # Number of last nodes to ignore when calculate RMSE
    init_offset = 5  # Number of initial nodes to ignore when calculate RMSE

    # --- RMSE --- #
    rmse_q = (
        np.sqrt(
            np.square(
                x_est[: biorbd_model.nbQ(), init_offset:-final_offset] - q_ref[:, init_offset : -final_offset - ns_mhe]
            ).mean(axis=1)
        ).mean()
        * 180
        / np.pi
    )
    std_q = (
        np.sqrt(
            np.square(
                x_est[: biorbd_model.nbQ(), init_offset:-final_offset] - q_ref[:, init_offset : -final_offset - ns_mhe]
            ).mean(axis=1)
        ).std()
        * 180
        / np.pi
    )
    rmse_f = np.sqrt(
        np.square(force_est[:, init_offset:-final_offset] - force_ref[:, init_offset : -final_offset - ns_mhe]).mean(
            axis=1
        )
    ).mean()
    std_f = np.sqrt(
        np.square(force_est[:, init_offset:-final_offset] - force_ref[:, init_offset : -final_offset - ns_mhe]).mean(
            axis=1
        )
    ).std()
    print(f"Q RMSE: {rmse_q} +/- {std_q}; F RMSE: {rmse_f} +/- {std_f}")
    x_ref = np.concatenate((x_ref, a_ref))

    return biorbd_model, {
        "x_est": x_est,
        "u_est": u_est,
        "x_ref": x_ref,
        "x_init": x_wt_noise,
        "u_ref": u_ref,
        "time_per_mhe": toc / ceil(ns / rt_ratio - ns_mhe),
        "time_tot": toc,
        "q_noise": q_noise,
        "N_mhe": ns_mhe,
        "N_tot": ns,
        "rt_ratio": rt_ratio,
        "f_est": force_est,
        "f_ref": force_ref,
    }


# Same offset used to compute RMSE
T = 8
Ns = 800
final_offset = 1
init_offset = 2

# Get data from MHE problem
model = "arm_wt_rot_scap.bioMod"
biorbd_model, mat_content = generate_data()

Ns_mhe = int(mat_content["N_mhe"])
ratio = int(mat_content["rt_ratio"])
x_est = mat_content["x_est"]
q_est = mat_content["x_est"][: biorbd_model.nbQ(), :]
dq_est = mat_content["x_est"][biorbd_model.nbQ() : biorbd_model.nbQ() * 2, :]
u_est = mat_content["u_est"]
f_est = mat_content["f_est"]
x_init = mat_content["x_init"]
x_ref = mat_content["x_ref"]
q_ref = mat_content["x_ref"]
u_ref = mat_content["u_ref"]
f_ref = mat_content["f_ref"]

# PLOT
seaborn.set_style("whitegrid")
seaborn.color_palette()
q_ref = q_ref[:, ::ratio]
x_init = x_init[:, ::ratio]
t_x = np.linspace(0, T, q_est.shape[1] - init_offset - final_offset)
t_u = np.linspace(0, T, u_est.shape[1] - init_offset - final_offset)

# ----- Plot Q -----#
size_police = 12
q_name = ["Glenohumeral plane of elevation", "Glenohumeral elevation", "Glenohumeral axial rotation", "Elbow flexion"]
fig = plt.figure("MHE_Results")
grid = plt.GridSpec(2, 4, wspace=0.15, hspace=0.4, left=0.06, right=0.99)
for i in [1, 3]:
    fig = plt.subplot(grid[0, :2]) if i == 1 else plt.subplot(grid[0, 2:])
    plt.xlabel("Time (s)", fontsize=size_police)
    if i == 1:
        plt.ylabel("Joint angle (°)", fontsize=size_police)
    plt.plot(t_x, x_est[i, init_offset:-final_offset] * 180 / np.pi)
    plt.plot(t_x, x_init[i, init_offset : -Ns_mhe - final_offset] * 180 / np.pi, alpha=0.8)
    plt.plot(t_x, q_ref[i, init_offset : -Ns_mhe - final_offset] * 180 / np.pi, alpha=0.8)
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    plt.title(q_name[i], fontsize=size_police)
    if i == 1:
        plt.legend(
            labels=["Estimation", "Reference", "Noisy reference"],
            bbox_to_anchor=(1.05, 1.25),
            loc="upper center",
            borderaxespad=0.0,
            ncol=3,
            frameon=False,
            fontsize=size_police,
        )

# ----- Plot muscle force -----#
muscles_names = ["Tri Long", "Delt Middle", "Infraspin", "Bic Short"]
fest_to_plot = f_est[[6, 13, 15, 18], :]
fref_to_plot = f_ref[[6, 13, 15, 18], :]
for i in range(len(muscles_names)):
    fig = plt.subplot(grid[1, i])
    plt.xlabel("Time (s)", fontsize=size_police)
    if i == 0:
        plt.ylabel("Muscle Force(N)", fontsize=size_police)
    plt.plot(t_u, fest_to_plot[i, init_offset:-final_offset])
    plt.plot(t_u, fref_to_plot[i, init_offset : -Ns_mhe - final_offset], alpha=0.8)
    plt.title(muscles_names[i], fontsize=size_police)
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
u_ref = u_ref[:, ::ratio]
plt.show()
