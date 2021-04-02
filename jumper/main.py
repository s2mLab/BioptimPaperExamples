from time import time

from jumper.plots import add_jumper_plots
from jumper.ocp import Jumper5Phases


if __name__ == "__main__":
    root_path = "/".join(__file__.split("/")[:-1]) + "/"
    model_paths = (
        root_path + "models/jumper2contacts.bioMod",
        root_path + "models/jumper1contacts.bioMod",
        root_path + "models/jumper1contacts.bioMod",
        root_path + "models/jumper1contacts.bioMod",
        root_path + "models/jumper2contacts.bioMod",
    )
    initial_pose = 0, 0, -0.5336, 1.4, 0.8, -0.9, 0.47
    n_shoot = 30, 15, 20, 30, 30
    time_min = 0.2, 0.05, 0.6, 0.05, 0.1
    time_max = 0.7, 0.5, 2.0, 0.5, 0.5
    phase_time = 0.3, 0.2, 0.6, 0.2, 0.2

    jumper = Jumper5Phases(model_paths, n_shoot, time_min, phase_time, time_max, initial_pose, n_thread=4)
    add_jumper_plots(jumper)

    tic = time()
    sol = jumper.solve(limit_memory_max_iter=200, exact_max_iter=1000)
    print(f"Time to solve : {time() - tic}sec")

    sol.print()
    sol.animate()
