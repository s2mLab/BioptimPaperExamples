"""
This is an example on how to use quaternion to represent the orientation of the root of the model.
The avatar must complete one somersault rotation while maximizing the twist rotation.
"""
import numpy as np
from bioptim import Solver
from somersault.ocp import prepare_ocp_quaternion, prepare_ocp


if __name__ == "__main__":
    root_folder = "/".join(__file__.split("/")[:-1])
    is_quaternion = False
    is_collocation = False
    np.random.seed(0)

    solver = Solver.IPOPT()
    if is_quaternion:
        ocp = prepare_ocp_quaternion(root_folder + "/models/JeChMesh_RootQuat.bioMod", final_time=1.5, n_shooting=100, is_collocation=is_collocation)
    else:
        ocp = prepare_ocp(root_folder + "/models/JeChMesh_8DoF.bioMod", final_time=1.5, n_shooting=400, is_collocation=is_collocation)
        if is_collocation:
            solver.set_convergence_tolerance(1e-2)
            solver.set_acceptable_constr_viol_tol(1e-2)

    solver.set_maximum_iterations(1000)
    sol = ocp.solve(solver)
    sol.animate(nb_frames=-1)
