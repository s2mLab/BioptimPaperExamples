from time import time

from bioptim import ControlType

from JumperOcp import JumperOcp, JumperCOLLOCATION


if __name__ == "__main__":
    root_path_model = "/".join(__file__.split("/")[:-1])
    jumper_model = JumperCOLLOCATION(root_path_model + "/models/")
    jumper = JumperOcp(jumper=jumper_model, control_type=ControlType.CONSTANT, n_phases=2)

    tic = time()
    sol = jumper.solve(
        limit_memory_max_iter=jumper_model.n_max_iter_limited_memory,
        exact_max_iter=jumper_model.n_max_iter_exact,
    )
    sol.print()
    print(f"Time to solve : {time() - tic}sec")
    sol.animate()
