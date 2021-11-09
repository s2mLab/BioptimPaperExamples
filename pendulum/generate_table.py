from time import time

import numpy as np
from bioptim import Solver, OdeSolver

from .pendulum.ocp import prepare_ocp


def generate_table(out):
    root_path = "/".join(__file__.split("/")[:-1])
    model_path = root_path + "/models/MassPoint_pendulum.bioMod"
    np.random.seed(0)

    # IPOPT
    for i, ode_solver in enumerate([OdeSolver.RK4(), OdeSolver.COLLOCATION()]):
        ocp = prepare_ocp(biorbd_model_path=model_path, ode_solver=ode_solver)
        solver = Solver.IPOPT()
        solver.set_linear_solver("ma57")
        solver.set_print_level(0)

        # --- Solve the program --- #
        tic = time()
        sol = ocp.solve(solver=solver)
        toc = time() - tic
        sol_merged = sol.merge_phases()

        out.solver.append(out.Solver("Ipopt"))
        out.solver[i].nx = sol_merged.states["all"].shape[0]
        out.solver[i].nu = sol_merged.controls["all"].shape[0]
        out.solver[i].ns = sol_merged.ns[0]
        out.solver[i].ode_solver = ode_solver
        out.solver[i].n_iteration = sol.iterations
        out.solver[i].cost = sol.cost
        out.solver[i].convergence_time = toc
        out.solver[i].compute_error_single_shooting(sol)
