from time import time

from bioptim import ControlType

from .JumperOcp import JumperOcp, Jumper, OdeSolver


def generate_table(out):
    root_path_model = "/".join(__file__.split("/")[:-1])

    for i, ode_solver in enumerate([OdeSolver.RK4(), OdeSolver.COLLOCATION()]):
        jumper_model = Jumper(root_path_model + "/models/")
        jumper = JumperOcp(jumper=jumper_model, control_type=ControlType.CONSTANT, n_phases=2, ode_solver=ode_solver)

        tic = time()
        sol = jumper.solve(limit_memory_max_iter=200, exact_max_iter=1000, force_no_graph=True, linear_solver="ma57")
        toc = time() - tic
        sol_merged = sol.merge_phases()

        out.solver.append(out.Solver("Ipopt"))
        out.solver[i].ode_solver = ode_solver
        out.solver[i].nx = sol_merged.states["all"].shape[0]
        out.solver[i].nu = sol_merged.controls["all"].shape[0]
        out.solver[i].ns = sol_merged.ns[0]
        out.solver[i].n_iteration = sol.iterations
        out.solver[i].cost = sol.cost
        out.solver[i].convergence_time = toc
        out.solver[i].compute_error_single_shooting(sol)
