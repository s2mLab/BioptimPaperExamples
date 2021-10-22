from time import time

import numpy as np
import biorbd_casadi as biorbd
from bioptim import Solver

from .pointing.ocp import prepare_ocp


def generate_table(out):
    model_path = "/".join(__file__.split("/")[:-1]) + "/models/arm26.bioMod"
    biorbd_model_ip = biorbd.Model(model_path)

    # IPOPT
    use_ipopt = True
    use_excitations = True
    use_collocation = False
    if use_excitations:
        weights = np.array([10, 1, 10, 100000, 1]) if not use_ipopt else np.array([10, 0.1, 10, 10000, 0.1])
    else:
        weights = np.array([100, 1, 1, 100000, 1]) if not use_ipopt else np.array([100, 1, 1, 100000, 1])
    ocp = prepare_ocp(
        biorbd_model=biorbd_model_ip,
        final_time=2,
        n_shooting=200,
        use_sx=not use_ipopt,
        weights=weights,
        use_excitations=use_excitations,
        use_collocation=use_collocation,
    )
    solver_ipopt = Solver.IPOPT()
    solver_ipopt.set_linear_solver("ma57")
    solver_ipopt.set_print_level(0)

    # --- Solve the program --- #
    tic = time()
    sol = ocp.solve(solver_ipopt)

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
    use_excitations = True
    biorbd_model_ac = biorbd.Model(model_path)

    ocp = prepare_ocp(
        biorbd_model=biorbd_model_ac,
        final_time=2,
        n_shooting=200,
        use_sx=not use_ipopt,
        weights=weights,
        use_excitations=use_excitations,
    )
    solver_acados = Solver.ACADOS()
    solver_acados.set_sim_method_num_steps(5)
    solver_acados.set_convergence_tolerance(1e-8)
    solver_acados.set_integrator_type("ERK")
    solver_acados.set_hessian_approx("GAUSS_NEWTON")
    solver_acados.set_print_level(0)

    # --- Solve the program --- #
    sol = ocp.solve(solver_acados)

    out.solver.append(out.Solver("Acados"))
    out.solver[1].n_iteration = sol.iterations
    out.solver[1].cost = sol.cost
    out.solver[1].convergence_time = sol.solver_time_to_optimize
    out.solver[1].compute_error_single_shooting(sol, 1)
