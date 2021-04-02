import pickle
from time import time

import biorbd
import numpy as np
from bioptim import (
    Solution,
    BoundsList,
    QAndQDotBounds,
    InitialGuess,
    Solver,
    InterpolationType,
)

from .mhe.ocp import prepare_ocp, prepare_short_ocp, generate_noise, define_objective, warm_start_mhe


def generate_table(out):
    root_path = "/".join(__file__.split("/")[:-1])
    model_path = root_path + "/models/arm_wt_rot_scap.bioMod"
    biorbd_model = biorbd.Model(model_path)

    # --- Prepare and solve MHE --- #
    t = 8
    ns = 800
    ns_mhe = 7
    rt_ratio = 3
    t_mhe = t / (ns / rt_ratio) * ns_mhe

    # --- Prepare reference data --- #
    with open(f"{root_path}/data/sim_ac_8000ms_800sn_REACH2_co_level_0_step5_ERK.bob", "rb") as file:
        data = pickle.load(file)
    states = data["data"][0]
    controls = data["data"][1]
    q_ref, dq_ref, u_ref = states["q"], states["qdot"], controls["muscles"]

    ocp = prepare_ocp(biorbd_model=biorbd_model, final_time=t_mhe, n_shooting=ns_mhe)
    q_noise = 5
    x_ref = np.concatenate((generate_noise(biorbd_model, q_ref, q_noise), dq_ref))
    x_est = np.zeros((biorbd_model.nbQ() * 2, x_ref[:, ::rt_ratio].shape[1] - ns_mhe))
    u_est = np.zeros((biorbd_model.nbMuscles(), u_ref[:, ::rt_ratio].shape[1] - ns_mhe))

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
            solver_options={"nlp_solver_tol_comp": 1e-6, "nlp_solver_tol_eq": 1e-6, "nlp_solver_tol_stat": 1e-5},
        )
        # Set solutions and set initial guess for next optimisation
        x0, u0, x_out, u_out = warm_start_mhe(sol)
        x_est[:, i] = x_out
        if i < u_est.shape[1]:
            u_est[:, i] = u_out

    toc = time() - tic
    n = x_est.shape[1] - 1
    tf = (ns - ns % rt_ratio) / (ns / t)
    final_time = tf - (ns_mhe * (tf / (n + ns_mhe)))
    short_ocp = prepare_short_ocp(biorbd_model, final_time=final_time, n_shooting=n)
    x_init_guess = InitialGuess(x_est, interpolation=InterpolationType.EACH_FRAME)
    u_init_guess = InitialGuess(u_est, interpolation=InterpolationType.EACH_FRAME)
    sol = Solution(short_ocp, [x_init_guess, u_init_guess])

    out.solver.append(out.Solver("Acados"))
    out.nx = x_est.shape[0]
    out.nu = u_est.shape[0]
    out.ns = n
    out.solver[0].n_iteration = "N.A."
    out.solver[0].cost = "N.A."
    out.solver[0].convergence_time = toc
    out.solver[0].compute_error_single_shooting(sol, 1)
