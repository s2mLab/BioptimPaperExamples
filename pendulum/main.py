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

    print("*********************************************")
    print(f"Time to solve : {toc}sec")
    print("*********************************************")
    sol.graphs()
