"""
This is a basic example on how to use moving horizon estimation for muscle force estimation using a 4 degree of freedom
(Dof) Arm model actuated by 19 hill-type muscles. controls are muscle activations.
Model joint angles are tracked to match with reference ones, muscle activations are minimized.
"""
from time import time
from math import ceil

import biorbd_casadi as biorbd
import numpy as np
import scipy.io as sio
import bioviz

from mhe.ocp import muscle_force_func, generate_noise, prepare_mhe, get_reference_data, update_mhe


# --- RMSE --- #
def rmse(data, data_ref):
    return np.sqrt(((data - data_ref) ** 2).mean())


if __name__ == "__main__":
    """
    Prepare and solve the MHE example
    """

    root_path = "/".join(__file__.split("/")[:-1]) + "/"
    model_path = root_path + "/models/arm_wt_rot_scap.bioMod"
    biorbd_model = biorbd.Model(model_path)

    # --- Prepare and solve MHE --- #
    use_noise = True  # True to add noise on reference joint angles
    q_noise = 3
    t = 8
    ns = 800
    ns_mhe = 7
    rt_ratio = 3
    t_mhe = t / (ns / rt_ratio) * ns_mhe

    # --- Prepare reference data --- #
    q_ref, dq_ref, act_ref = get_reference_data(f"{root_path}/data/sim_ac_8000ms_800sn_REACH2_co_level_0_step5_ERK.bob")
    x_ref_no_noise = np.concatenate((q_ref, dq_ref))
    x_ref = np.concatenate((generate_noise(biorbd_model, q_ref, q_noise), dq_ref) if use_noise else (q_ref, dq_ref))

    # Initialize MHE
    mhe, solver = prepare_mhe(biorbd_model=biorbd_model, final_time=t_mhe, n_shooting=ns_mhe, x_ref=x_ref, rt_ratio=rt_ratio)
    final_time_index = x_ref[:, ::rt_ratio].shape[1] - ns_mhe

    # Solve the program
    tic = time()  # Save initial time
    sol = mhe.solve(
        lambda mhe, i, sol: update_mhe(mhe, i, sol, q_ref, ns_mhe, rt_ratio, final_time_index), solver=solver
    )
    toc = time() - tic

    # Show some statistics
    q_est, dq_est, act_est = sol.states["q"], sol.states["qdot"], sol.controls["muscles"]

    muscle_force = muscle_force_func(biorbd_model)
    force_est = np.array(muscle_force(q_est, dq_est, [], act_est))
    force_ref = np.array(muscle_force(q_ref, dq_ref, [], act_ref))

    final_offset = 5  # Number of last nodes to ignore when calculate RMSE
    init_offset = 5  # Number of initial nodes to ignore when calculate RMSE
    offset = ns_mhe

    to_deg = 180 / np.pi
    q_ref = q_ref[:, ::rt_ratio]
    rmse_q = rmse(q_est[:, init_offset:-final_offset], q_ref[:, init_offset : -final_offset - ns_mhe]) * to_deg
    std_q = np.std(q_est[: , init_offset:-final_offset] - q_ref[:, init_offset : -final_offset - ns_mhe]) * to_deg

    force_ref = force_ref[:, ::rt_ratio]
    rmse_f = rmse(force_est[:, init_offset:-final_offset], force_ref[:, init_offset : -final_offset - ns_mhe])
    std_f = np.std(force_est[:, init_offset:-final_offset] - force_ref[:, init_offset : -final_offset - ns_mhe])
    print(f"Q RMSE: {rmse_q} +/- {std_q}; F RMSE: {rmse_f} +/- {std_f}")
    x_ref = np.concatenate((x_ref, act_ref))

    dic = {
        "x_est": sol.states["all"],
        "u_est": sol.controls["all"],
        "x_ref": x_ref,
        "x_init": x_ref_no_noise,
        "u_ref": act_ref,
        "time_per_mhe": toc / ceil(ns / rt_ratio - ns_mhe),
        "time_tot": toc,
        "q_noise": q_noise,
        "N_mhe": ns_mhe,
        "N_tot": ns,
        "rt_ratio": rt_ratio,
        "f_est": force_est,
        "f_ref": force_ref,
    }
    sio.savemat(f"data/MHE_results.mat", dic)
    duration = 1
    t = 8
    ns = 800

    print("*********************************************")
    print(f"Problem solved with Acados")
    print(f"Solving time : {dic['time_tot']}s")
    print(f"Solving frequency : {1/dic['time_per_mhe']}s")

    # ------ Animate ------ #
    b = bioviz.Viz(model_path)
    b.load_movement(q_est)
    b.exec()
