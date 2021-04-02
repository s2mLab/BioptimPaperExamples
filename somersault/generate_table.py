from time import time
from numpy import random

from .somersault.ocp import prepare_ocp, prepare_ocp_quaternion


def generate_table(out):
    random.seed(0)
    root_folder = "/".join(__file__.split("/")[:-1])
    ocp_euler = prepare_ocp(root_folder + "/models/JeChMesh_8DoF.bioMod", final_time=1.5, n_shooting=100)
    ocp_quaternion = prepare_ocp_quaternion(
        root_folder + "/models/JeChMesh_RootQuat.bioMod", final_time=1.5, n_shooting=100
    )

    # --- Solve the program --- #
    tic = time()
    sol_euler = ocp_euler.solve(
        solver_options={"max_iter": 1000, "linear_solver": "ma57"}
    )
    toc_euler = time() - tic
    tic = time()
    sol_quaternion = ocp_quaternion.solve(
        solver_options={"max_iter": 1000, "linear_solver": "ma57"}
    )
    toc_quaternion = time() - tic

    out.nx = sol_euler.states["all"].shape[0]
    out.nu = sol_euler.controls["all"].shape[0]
    out.ns = sol_euler.ns[0]

    out.solver.append(out.Solver("Euler"))
    out.solver[0].n_iteration = sol_euler.iterations
    out.solver[0].cost = sol_euler.cost
    out.solver[0].convergence_time = toc_euler
    out.solver[0].compute_error_single_shooting(sol_euler, 1)

    out.solver.append(out.Solver("Quaternion"))
    out.solver[1].n_iteration = sol_quaternion.iterations
    out.solver[1].cost = sol_quaternion.cost
    out.solver[1].convergence_time = toc_quaternion
    out.solver[1].compute_error_single_shooting(sol_quaternion, 1)
