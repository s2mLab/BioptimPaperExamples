from time import time

from bioptim import ControlType

from .JumperOcp import JumperOcp, JumperCOLLOCATION, JumperRK4


def generate_table(out):
    root_path_model = "/".join(__file__.split("/")[:-1])

    for i, jumper_type in enumerate([JumperRK4, JumperCOLLOCATION]):
        jumper_model = jumper_type(root_path_model + "/models/")
        jumper = JumperOcp(jumper=jumper_model, control_type=ControlType.CONSTANT, n_phases=2)

        tic = time()
        sol = jumper.solve(
            limit_memory_max_iter=jumper_model.n_max_iter_limited_memory,
            exact_max_iter=jumper_model.n_max_iter_exact,
            force_no_graph=True,
            linear_solver="ma57"
        )
        toc = time() - tic
        sol_merged = sol.merge_phases()

        out.solver.append(out.Solver("Ipopt"))
        out.solver[i].ode_solver = jumper_model.ode_solver
        out.solver[i].nx = sol_merged.states["all"].shape[0]
        out.solver[i].nu = sol_merged.controls["all"].shape[0]
        out.solver[i].ns = sol_merged.ns[0]
        out.solver[i].n_iteration = sol.iterations
        out.solver[i].cost = sol.cost
        out.solver[i].convergence_time = toc
        out.solver[i].compute_error_single_shooting(sol)
