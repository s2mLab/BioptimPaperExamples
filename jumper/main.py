from time import time

from bioptim import ControlType

from JumperOcp import JumperOcp, Jumper

if __name__ == "__main__":
    root_path_model = "/".join(__file__.split("/")[:-1])
    jumper_model = Jumper(root_path_model + "/models/")
    jumper = JumperOcp(jumper=jumper_model, control_type=ControlType.CONSTANT, n_phases=3)

    tic = time()
    sol = jumper.solve(limit_memory_max_iter=400, exact_max_iter=1000)
    print(f"Time to solve : {time() - tic}sec")

    sol.print()
    sol.animate()
