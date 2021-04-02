from time import time

import numpy as np
import biorbd
from bioptim import Solver

from .pointing.ocp import prepare_ocp


def generate_table(out):
    model_path = "/".join(__file__.split("/")[:-1]) + "/models/arm26.bioMod"
    biorbd_model_ip = biorbd.Model(model_path)

    # IPOPT
    use_ipopt = True
    weights = np.array([100, 1, 1, 100000])
    ocp = prepare_ocp(biorbd_model=biorbd_model_ip, final_time=2, n_shooting=50, use_sx=not use_ipopt, weights=weights)
    opts = {"linear_solver": "ma57", "hessian_approximation": "exact"}
    solver = Solver.IPOPT

    # --- Solve the program --- #
    tic = time()
    sol = ocp.solve(
        solver=solver,
        solver_options=opts,
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
    out.solver[0].compute_error_single_shooting(sol, 1)

    # ACADOS
    use_ipopt = False
    biorbd_model_ac = biorbd.Model(model_path)
    ocp = prepare_ocp(biorbd_model=biorbd_model_ac, final_time=2, n_shooting=50, use_sx=not use_ipopt, weights=weights)
    opts = {"sim_method_num_steps": 5, "tol": 1e-8, "integrator_type": "ERK", "hessian_approx": "GAUSS_NEWTON"}
    solver = Solver.ACADOS

    # --- Solve the program --- #
    sol = ocp.solve(
        solver=solver,
        solver_options=opts,
    )

    out.solver.append(out.Solver("Acados"))
    out.solver[1].n_iteration = sol.iterations
    out.solver[1].cost = sol.cost
    out.solver[1].convergence_time = sol.time_to_optimize
    out.solver[1].compute_error_single_shooting(sol, 1)
