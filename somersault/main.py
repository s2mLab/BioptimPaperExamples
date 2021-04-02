"""
This is an example on how to use quaternion to represent the orientation of the root of the model.
The avatar must complete one somersault rotation while maximizing the twist rotation.
"""
import numpy as np

from somersault.ocp import prepare_ocp, prepare_ocp_quaternion


if __name__ == "__main__":
    root_folder = "/".join(__file__.split("/")[:-1])
    is_quaternion = True
    np.random.seed(0)

    if is_quaternion:
        ocp = prepare_ocp_quaternion(root_folder + "/models/JeChMesh_RootQuat.bioMod", final_time=1.5, n_shooting=100)
    else:
        ocp = prepare_ocp(root_folder + "/models/JeChMesh_8DoF.bioMod", final_time=1.5, n_shooting=100)

    sol = ocp.solve(solver_options={"tol": 1e-15, "constr_viol_tol": 1e-15, "max_iter": 1000})
    sol.animate()
