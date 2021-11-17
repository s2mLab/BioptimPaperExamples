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


if __name__ == "__main__":
    """
    Prepare and solve and animate a reaching task ocp
    """
    use_ipopt = False
    use_excitations = True
    use_collocation = True
    if use_excitations:
        weights = np.array([10, 5, 10, 100000, 1]) if not use_ipopt else np.array([10, 0.1, 10, 10000, 0.1])
    else:
        weights = np.array([100, 1, 1, 100000, 1]) if not use_ipopt else np.array([10, 1, 1, 100000, 1])

    model_path = "/".join(__file__.split("/")[:-1]) + "/models/arm26.bioMod"
    biorbd_model = biorbd.Model(model_path)
    ode_solver = OdeSolver.COLLOCATION() if (use_collocation and use_ipopt) else OdeSolver.RK4()
    if use_collocation:
        n_shooting = 120
    else:
        n_shooting = 100
    ocp = prepare_ocp(
        biorbd_model=biorbd_model,
        final_time=1,
        n_shooting=n_shooting,
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
    print("*********************************************")
    print(f"Problem solved with {solver.type}")
    if use_ipopt:
        print(f"Trasncription method: {ode_solver.__str__()}")
    else:
        print(f"Transcription method: multiple shooting")
    print(f"Solving time : {sol.solver_time_to_optimize}s")

    sol.animate(
        show_meshes=True,
        background_color=(1, 1, 1),
        show_local_ref_frame=False,
        show_global_center_of_mass=False,
        show_segments_center_of_mass=False,
        show_global_ref_frame=False,
    )
