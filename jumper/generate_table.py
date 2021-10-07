from time import time

from bioptim import ControlType

from JumperOcp import JumperOcp, Jumper


def generate_table(out):
    root_path_model = "/".join(__file__.split("/")[:-1])
    jumper_model = Jumper(root_path_model + "/models/")
    jumper = JumperOcp(jumper=jumper_model, control_type=ControlType.CONSTANT, n_phases=5)

    tic = time()
    sol = jumper.solve(limit_memory_max_iter=2, exact_max_iter=2, force_no_graph=True, linear_solver="ma57")
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

out = []
generate_table(out)