"""
This is a basic code to plot and animate the results of MHE example.

Please note that before to use this code you have to run main.py to generate results.
"""
from time import time
import seaborn
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import biorbd_casadi as biorbd
import numpy as np

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
    np.random.seed(450)
    use_noise = True  # True to add noise on reference joint angles
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

    x_ref_no_noise = np.concatenate((q_ref_no_noise, dq_ref_no_noise))
    x_ref = np.concatenate(
        (generate_noise(biorbd_model, q_ref_no_noise, q_noise_lvl), dq_ref_no_noise)
        if use_noise
        else (q_ref_no_noise, dq_ref_no_noise)
    )

    q_ref, dq_ref = x_ref[: biorbd_model.nbQ(), :], x_ref[biorbd_model.nbQ() : biorbd_model.nbQ() * 2, :]

    # Initialize MHE
    mhe, solver_options = prepare_mhe(
        biorbd_model=biorbd_model,
        final_time=t_mhe,
        n_shooting=ns_mhe,
        x_ref=x_ref,
        rt_ratio=rt_ratio,
        use_noise=use_noise,
    )

    final_time_index = x_ref[:, ::rt_ratio].shape[1] - ns_mhe

    # Solve the program
    tic = time()  # Save initial time
    sol = mhe.solve(
        lambda mhe, i, sol: update_mhe(mhe, i, sol, q_ref, ns_mhe, rt_ratio, final_time_index),
        solver_options=solver_options,
    )

    # sol.graphs()
    toc = time() - tic

    # Show some statistics
    q_est, dq_est, muscle_controls_est = sol.states["q"], sol.states["qdot"], sol.controls["muscles"]
    muscle_controls_ref = act_ref_no_noise
    muscle_force = muscle_force_func(biorbd_model)
    force_est = np.array(muscle_force(q_est, dq_est, [], muscle_controls_est))
    force_ref = np.array(muscle_force(q_ref_no_noise, dq_ref_no_noise, [], muscle_controls_ref))
    final_offset = 5  # Number of last nodes to ignore when calculate RMSE
    init_offset = 5  # Number of initial nodes to ignore when calculate RMSE
    offset = ns_mhe

    to_deg = 180 / np.pi
    q_ref = q_ref[:, ::rt_ratio]
    rmse_q = rmse(q_est[:, init_offset:-final_offset], q_ref[:, init_offset : -final_offset - ns_mhe]) * to_deg
    std_q = np.std(q_est[:, init_offset:-final_offset] - q_ref[:, init_offset : -final_offset - ns_mhe]) * to_deg

    force_ref = force_ref[:, ::rt_ratio]
    rmse_f = rmse(force_est[:, init_offset:-final_offset], force_ref[:, init_offset : -final_offset - ns_mhe])
    std_f = np.std(force_est[:, init_offset:-final_offset] - force_ref[:, init_offset : -final_offset - ns_mhe])
    print(f"Q RMSE: {rmse_q} +/- {std_q}; F RMSE: {rmse_f} +/- {std_f}")
    # x_ref = np.concatenate((x_ref, act_ref))

    # Same offset used to compute RMSE
    T = 8
    Ns = 800

    # Get data from MHE problem
    model = "arm_wt_rot_scap.bioMod"

    # PLOT
    seaborn.set_style("whitegrid")
    seaborn.color_palette()
    x_est = np.concatenate((q_est, dq_est))
    x_ref = x_ref[:, ::rt_ratio]
    x_init = x_ref_no_noise[:, ::rt_ratio]
    t_x = np.linspace(0, T, q_est.shape[1] - init_offset - final_offset)
    t_u = np.linspace(0, T, muscle_controls_est.shape[1] - init_offset - final_offset)

    # ----- Plot Q -----#
    size_police = 12
    q_name = [
        "Glenohumeral plane of elevation",
        "Glenohumeral elevation",
        "Glenohumeral axial rotation",
        "Elbow flexion",
    ]
    fig = plt.figure("MHE_Results")
    grid = plt.GridSpec(2, 4, wspace=0.15, hspace=0.4, left=0.06, right=0.99)
    for i in [1, 3]:
        fig = plt.subplot(grid[0, :2]) if i == 1 else plt.subplot(grid[0, 2:])
        plt.xlabel("Time (s)", fontsize=size_police)
        if i == 1:
            plt.ylabel("Joint angle (°)", fontsize=size_police)
        plt.plot(t_x, x_est[i, init_offset:-final_offset] * 180 / np.pi)
        plt.plot(t_x, x_init[i, init_offset : -ns_mhe - final_offset] * 180 / np.pi, alpha=0.8)
        plt.plot(t_x, x_ref[i, init_offset : -ns_mhe - final_offset] * 180 / np.pi, alpha=0.8)
        plt.gca().set_prop_cycle(None)
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
    muscles_names = ["Tri long", "Delt middle", "Infraspin", "Bic short"]
    fest_to_plot = force_est[[6, 13, 15, 18], :]
    fref_to_plot = force_ref[[6, 13, 15, 18], :]
    for i in range(len(muscles_names)):
        fig = plt.subplot(grid[1, i])
        plt.xlabel("Time (s)", fontsize=size_police)
        if i == 0:
            plt.ylabel("Muscle force (N)", fontsize=size_police)
        plt.plot(t_u, fest_to_plot[i, init_offset:-final_offset])
        plt.plot(t_u, fref_to_plot[i, init_offset : -ns_mhe - final_offset], alpha=0.8)
        plt.title(muscles_names[i], fontsize=size_police)
        plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    u_ref = muscle_controls_est[:, ::rt_ratio]
    plt.show()
