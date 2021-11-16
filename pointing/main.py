"""
This is a basic example on how to use biorbd model driven by muscle to perform an optimal reaching task.
The arms must reach a marker placed upward in front while minimizing the muscles activity

Please note that using show_meshes=True in the animator may be long due to the creation of a huge CasADi graph of the
mesh points.
"""

import biorbd_casadi as biorbd
import numpy as np
from bioptim import Solver, Shooting, OdeSolver
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

    sol_int = sol.integrate(
        shooting_type=Shooting.SINGLE_CONTINUOUS,
        merge_phases=True,
        keep_intermediate_points=False,
        use_scipy_integrator=True,
    )
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
    use_ipopt = True
    use_excitations = True
    use_collocation = True
    if use_excitations:
        weights = np.array([10, 5, 10, 100000, 1]) if not use_ipopt else np.array([10, 0.1, 10, 10000, 0.1])
    else:
        weights = np.array([100, 1, 1, 100000, 1]) if not use_ipopt else np.array([10, 1, 1, 100000, 1])

    model_path = "/".join(__file__.split("/")[:-1]) + "/models/arm26.bioMod"
    biorbd_model = biorbd.Model(model_path)
    ode_solver = OdeSolver.COLLOCATION() if use_collocation else OdeSolver.RK4()
    if use_collocation:
        n_shooting = 120
    else:
        n_shooting = 100
    ocp = prepare_ocp(
        biorbd_model=biorbd_model,
        final_time=1,
        n_shooting=nshooting,
        use_sx=not use_ipopt,
        weights=weights,
        use_excitations=use_excitations,
        ode_solver=ode_solver,
    )

    # --- Solve the program --- #
    if use_ipopt:
        solver = Solver.IPOPT()
        solver.set_hessian_approximation("exact")
    else:
        solver = Solver.ACADOS()
        solver.set_sim_method_num_steps(5)
        solver.set_convergence_tolerance(1e-6)
        solver.set_integrator_type("ERK")
        solver.set_hessian_approx("GAUSS_NEWTON")
        solver.set_maximum_iterations(1000)
    sol = ocp.solve(solver=solver)

    # --- Show results --- #
    single_shooting_duration = 1
    ss_err_t, ss_err_r = compute_error_single_shooting(sol, 1)

    print("*********************************************")
    print(f"Problem solved with {solver.type}")
    print(f"Solving time : {sol.solver_time_to_optimize}s")
    print(f"Single shooting error at {single_shooting_duration}s in translation (mm)= {ss_err_t}")
    print(f"Single shooting error at {single_shooting_duration}s in rotation (°)= {ss_err_r}")
    sol.graphs()
    sol.animate(
        show_meshes=True,
        background_color=(1, 1, 1),
        show_local_ref_frame=False,
        show_global_center_of_mass=False,
        show_segments_center_of_mass=False,
        show_global_ref_frame=False,
    )
