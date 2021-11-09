from time import time
from numpy import random
from bioptim import Solver

from .somersault.ocp import prepare_ocp, prepare_ocp_quaternion


def generate_table(out):
    random.seed(0)
    root_folder = "/".join(__file__.split("/")[:-1])
    ocp_euler_RK4 = prepare_ocp(root_folder + "/models/JeChMesh_8DoF.bioMod", final_time=1.5, n_shooting=100, is_collocation=False)
    ocp_euler_COLLOCATIONS = prepare_ocp(root_folder + "/models/JeChMesh_8DoF.bioMod", final_time=1.5, n_shooting=100, is_collocation=True)
    ocp_quaternion_RK4 = prepare_ocp_quaternion(
        root_folder + "/models/JeChMesh_RootQuat.bioMod", final_time=1.5, n_shooting=100, is_collocation=False
    )

    # --- Solve the program --- #
    solver = Solver.IPOPT()
    solver.set_print_level(0)

    # Euler RK4
    tic = time()
    sol_euler_RK4 = ocp_euler_RK4.solve(solver)
    toc_euler_RK4 = time() - tic

    # Quaternion
    tic = time()
    sol_quaternion_RK4 = ocp_quaternion_RK4.solve(solver)
    toc_quaternion_RK4 = time() - tic

    # Euler collocations
    solver.set_convergence_tolerance(1e-2)
    solver.set_acceptable_constr_viol_tol(1e-2)
    tic = time()
    sol_euler_COLLOCATIONS = ocp_euler_COLLOCATIONS.solve(solver)
    toc_euler_COLLOCATIONS = time() - tic

    out.solver.append(out.Solver("Euler RK4"))
    out.solver[0].nx = sol_euler_RK4.states["all"].shape[0]
    out.solver[0].nu = sol_euler_RK4.controls["all"].shape[0]
    out.solver[0].ns = sol_euler_RK4.ns[0]
    out.solver[0].n_iteration = sol_euler_RK4.iterations
    out.solver[0].cost = sol_euler_RK4.cost
    out.solver[0].convergence_time = toc_euler_RK4
    out.solver[0].compute_error_single_shooting(sol_euler_RK4, 1)

    out.solver.append(out.Solver("Euler Collocations"))
    out.solver[1].nx = sol_euler_COLLOCATIONS.states["all"].shape[0]
    out.solver[1].nu = sol_euler_COLLOCATIONS.controls["all"].shape[0]
    out.solver[1].ns = sol_euler_COLLOCATIONS.ns[0]
    out.solver[1].n_iteration = sol_euler_COLLOCATIONS.iterations
    out.solver[1].cost = sol_euler_COLLOCATIONS.cost
    out.solver[1].convergence_time = toc_euler_COLLOCATIONS
    out.solver[1].compute_error_single_shooting(sol_euler_COLLOCATIONS, 1)

    out.solver.append(out.Solver("Quaternion RK4"))
    out.solver[2].nx = sol_quaternion_RK4.states["all"].shape[0]
    out.solver[2].nu = sol_quaternion_RK4.controls["all"].shape[0]
    out.solver[2].ns = sol_quaternion_RK4.ns[0]
    out.solver[2].n_iteration = sol_quaternion_RK4.iterations
    out.solver[2].cost = sol_quaternion_RK4.cost
    out.solver[2].convergence_time = toc_quaternion_RK4
    out.solver[2].compute_error_single_shooting(sol_quaternion_RK4, 1)
