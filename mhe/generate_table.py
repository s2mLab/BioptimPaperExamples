import biorbd_casadi as biorbd
import numpy as np
from bioptim import Solution, InitialGuess, InterpolationType, OdeSolver

from .mhe.ocp import get_reference_data, prepare_mhe, prepare_short_ocp, generate_noise, update_mhe


def generate_table(out):
    root_path = "/".join(__file__.split("/")[:-1]) + "/"
    model_path = root_path + "/models/arm_wt_rot_scap.bioMod"
    biorbd_model = biorbd.Model(model_path)

    # --- Prepare and solve MHE --- #
    np.random.seed(450)
    use_noise = False  # True to add noise on reference joint angles
    q_noise_lvl = 4
    t = 8
    ns = 800
    ns_mhe = 7
    rt_ratio = 3
    t_mhe = t / (ns / rt_ratio) * ns_mhe

    # --- Prepare reference data --- #
    q_ref_no_noise, dq_ref_no_noise, act_ref_no_noise, exc_ref_no_noise = get_reference_data(
        f"{root_path}/data/sim_ac_8000ms_800sn_REACH2_co_level_0_step5_ERK.bob"
    )

    x_ref = np.concatenate(
        (generate_noise(biorbd_model, q_ref_no_noise, q_noise_lvl), dq_ref_no_noise)
        if use_noise
        else (q_ref_no_noise, dq_ref_no_noise)
    )

    q_ref, dq_ref = x_ref[: biorbd_model.nbQ(), :], x_ref[biorbd_model.nbQ() : biorbd_model.nbQ() * 2, :]

    # Initialize MHE
    mhe, solver = prepare_mhe(
        biorbd_model=biorbd_model,
        final_time=t_mhe,
        n_shooting=ns_mhe,
        x_ref=x_ref,
        rt_ratio=rt_ratio,
        use_noise=use_noise,
    )

    final_time_index = x_ref[:, ::rt_ratio].shape[1] - ns_mhe

    # Solve the program
    sol = mhe.solve(
        lambda mhe, i, sol: update_mhe(mhe, i, sol, q_ref, ns_mhe, rt_ratio, final_time_index), solver=solver
    )
    time_to_optimize = sol.solver_time_to_optimize

    # Compute some statistics
    final_time_index -= 1
    tf = (ns - ns % rt_ratio) / (ns / t)
    final_time = tf - (ns_mhe * (tf / (final_time_index + ns_mhe)))
    short_ocp = prepare_short_ocp(model_path, final_time=final_time, n_shooting=final_time_index)
    x_init_guess = InitialGuess(sol.states["all"], interpolation=InterpolationType.EACH_FRAME)
    u_init_guess = InitialGuess(sol.controls["all"], interpolation=InterpolationType.EACH_FRAME)
    sol = Solution(short_ocp, [x_init_guess, u_init_guess])

    out.solver.append(out.Solver("Acados"))
    out.solver[0].ode_solver = OdeSolver.RK4()
    out.solver[0].nx = final_time_index
    out.solver[0].nu = final_time_index - 1
    out.solver[0].ns = final_time_index
    out.solver[0].n_iteration = "N.A."
    out.solver[0].cost = "N.A."
    out.solver[0].convergence_time = time_to_optimize
    out.solver[0].compute_error_single_shooting(sol, 1)
