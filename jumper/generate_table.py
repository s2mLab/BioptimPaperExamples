from time import time

from .jumper.ocp import Jumper5Phases


def generate_table(out):
    root_path_model = "/".join(__file__.split("/")[:-1]) + "/models/"
    model_paths = (
        root_path_model + "jumper2contacts.bioMod",
        root_path_model + "jumper1contacts.bioMod",
        root_path_model + "jumper1contacts.bioMod",
        root_path_model + "jumper1contacts.bioMod",
        root_path_model + "jumper2contacts.bioMod",
    )
    initial_pose = 0, 0, -0.5336, 1.4, 0.8, -0.9, 0.47
    n_shoot = 30, 15, 20, 30, 30
    time_min = 0.2, 0.05, 0.6, 0.05, 0.1
    time_max = 0.7, 0.5, 2.0, 0.5, 0.5
    phase_time = 0.3, 0.2, 0.6, 0.2, 0.2
    jumper = Jumper5Phases(model_paths, n_shoot, time_min, phase_time, time_max, initial_pose, n_thread=8)

    tic = time()
    sol = jumper.solve(limit_memory_max_iter=200, exact_max_iter=1000, force_no_graph=True, linear_solver="ma57")
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
