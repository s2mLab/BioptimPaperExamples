"""
This is a basic example on how to use biorbd model driven by muscle to perform an optimal reaching task.
The arms must reach a marker placed upward in front while minimizing the muscles activity

Please note that using show_meshes=True in the animator may be long due to the creation of a huge CasADi graph of the
mesh points.
"""

import biorbd
import numpy as np
from bioptim import Solver, Shooting

from pointing.ocp import prepare_ocp


def compute_error_single_shooting(sol, duration):
    sol_merged = sol.merge_phases()

    if sol_merged.phase_time[-1] < duration:
        raise ValueError(
            f"Single shooting integration duration must be smaller than ocp duration :{sol_merged.phase_time[-1]} s"
        )

    trans_idx = []
    rot_idx = []
    for i in range(sol.ocp.nlp[0].model.nbQ()):
        if sol.ocp.nlp[0].model.nameDof()[i].to_string()[-4:-1] == "Rot":
            rot_idx += [i]
        else:
            trans_idx += [i]
    rot_idx = np.array(rot_idx)
    trans_idx = np.array(trans_idx)

    sol_int = sol.integrate(shooting_type=Shooting.SINGLE_CONTINUOUS, merge_phases=True, keepdims=True)
    sn_1s = int(sol_int.ns[0] / sol_int.phase_time[-1] * duration)  # shooting node at {duration} second
    if len(rot_idx) > 0:
        single_shoot_error_r = (
            np.sqrt(np.mean((sol_int.states["q"][rot_idx, sn_1s] - sol_merged.states["q"][rot_idx, sn_1s]) ** 2))
            * 180
            / np.pi
        )
    else:
        single_shoot_error_r = "N.A."
    if len(trans_idx) > 0:
        single_shoot_error_t = (
            np.sqrt(
                np.mean((sol_int.states["q"][trans_idx, 5 * sn_1s] - sol_merged.states["q"][trans_idx, sn_1s]) ** 2)
            )
            / 1000
        )
    else:
        single_shoot_error_t = "N.A."
    return single_shoot_error_t, single_shoot_error_r


if __name__ == "__main__":
    """
    Prepare and solve and animate a reaching task ocp
    """
    use_ipopt = False
    weights = np.array([100, 1, 1, 100000])
    model_path = "/".join(__file__.split("/")[:-1]) + "/models/arm26.bioMod"
    biorbd_model = biorbd.Model(model_path)
    ocp = prepare_ocp(biorbd_model=biorbd_model, final_time=2, n_shooting=50, use_sx=not use_ipopt, weights=weights)

    # --- Solve the program --- #
    if use_ipopt:
        opts = {"linear_solver": "mumps", "hessian_approximation": "exact"}
        solver = Solver.IPOPT
    else:
        opts = {"sim_method_num_steps": 5, "tol": 1e-8, "integrator_type": "ERK", "hessian_approx": "GAUSS_NEWTON"}
        solver = Solver.ACADOS
    sol = ocp.solve(solver=solver, solver_options=opts, show_online_optim=False)

    # --- Show results --- #
    sol.print()
    single_shooting_duration = 1
    ss_err_t, ss_err_r = compute_error_single_shooting(sol, 1)
    print("*********************************************")
    print(f"Problem solved with {solver.value}")
    print(f"Solving time : {sol.time_to_optimize}s")
    print(f"Single shooting error at {single_shooting_duration}s in translation (mm)= {ss_err_t}")
    print(f"Single shooting error at {single_shooting_duration}s in rotation (°)= {ss_err_r}")
    # result.graphs()
    sol.animate(
        show_meshes=True,
        background_color=(1, 1, 1),
        show_local_ref_frame=False,
        show_global_center_of_mass=False,
        show_segments_center_of_mass=False,
    )
