from time import time

import numpy as np
import biorbd_casadi as biorbd
from bioptim import Solver, OdeSolver

from .pointing.ocp import prepare_ocp


def generate_table(out):
    model_path = "/".join(__file__.split("/")[:-1]) + "/models/arm26.bioMod"
    i = 0
    for solver_name in ["Ipopt", "Acados"]:
        use_ipopt = solver_name == "Ipopt"
        use_excitations = True
        if use_excitations:
            weights = np.array([10, 5, 10, 100000, 1]) if not use_ipopt else np.array([10, 0.1, 10, 10000, 0.1])
        else:
            weights = np.array([100, 1, 1, 100000, 1]) if not use_ipopt else np.array([100, 1, 1, 100000, 1])

        for ode_solver in [OdeSolver.RK4(), OdeSolver.COLLOCATION()]:
            if not use_ipopt and isinstance(ode_solver, OdeSolver.COLLOCATION):
                continue
            if isinstance(ode_solver, OdeSolver.COLLOCATION):
                n_shooting = 120
            else:
                n_shooting = 100
            biorbd_model_ip = biorbd.Model(model_path)
            ocp = prepare_ocp(
                biorbd_model=biorbd_model_ip,
                final_time=1,
                n_shooting=n_shooting,
                use_sx=not use_ipopt,
                weights=weights,
                use_excitations=use_excitations,
                ode_solver=ode_solver
            )
            if use_ipopt:
                solver = Solver.IPOPT()
                solver.set_linear_solver("ma57")
                solver.set_hessian_approximation("exact")
                solver.set_print_level(0)

            else:
                solver = Solver.ACADOS()
                solver.set_sim_method_num_steps(5)
                solver.set_convergence_tolerance(1e-6)
                solver.set_maximum_iterations(1000)
                solver.set_integrator_type("ERK")
                solver.set_hessian_approx("GAUSS_NEWTON")
                solver.set_print_level(0)

            # --- Solve the program --- #
            tic = time()
            sol = ocp.solve(solver)

            toc = time() - tic
            sol_merged = sol.merge_phases()
            out.solver.append(out.Solver(solver_name))
            out.solver[i].nx = sol_merged.states["all"].shape[0]
            out.solver[i].nu = sol_merged.controls["all"].shape[0]
            out.solver[i].ns = sol_merged.ns[0]
            out.solver[i].ode_solver = ode_solver
            out.solver[i].n_iteration = sol.iterations
            out.solver[i].cost = sol.cost
            out.solver[i].convergence_time = toc
            out.solver[i].compute_error_single_shooting(sol)
            i += 1

