"""
This is a basic example on how to use external forces to model a spring.
The mass attached to the spring must stabilize its position during the second phase of the movement while perturbed by
the oscillation of a pendulum.
"""
from time import time

import numpy as np
from bioptim import Shooting, Solver

from pendulum.ocp import prepare_ocp

if __name__ == "__main__":
    root_path = "/".join(__file__.split("/")[:-1]) + "/"
    model_path = root_path + "models/MassPoint_pendulum.bioMod"
    np.random.seed(0)

    ocp = prepare_ocp(biorbd_model_path=model_path)

    # --- Solve the program --- #
    tic = time()
    sol = ocp.solve()
    toc = time() - tic

    def compute_error_single_shooting(ocp, sol, duration):
        if ocp.nlp[0].tf < duration:
            raise ValueError(
                f"Single shooting integration duration must be smaller than ocp duration :{ocp.nlp[0].tf} s"
            )
        sol_int = sol.integrate(shooting_type=Shooting.SINGLE_CONTINUOUS, use_scipy_integrator=True)
        sn_1s = int(ocp.nlp[0].ns / ocp.nlp[0].tf * duration)  # shooting node at {duration} second
        return np.sqrt(np.mean((sol_int.states[0]["all"][:, 5 * sn_1s] - sol.states[0]["all"][:, sn_1s]) ** 2))

    print("*********************************************")
    print(f"Single shooting error : {compute_error_single_shooting(ocp, sol, 1)}")
    print(f"Time to solve : {toc}sec")
    print("*********************************************")
    sol.graphs()
