from gait.generate_table import generate_table as gait_table
from jumper.generate_table import generate_table as jumper_table
from mhe.generate_table import generate_table as mhe_table
from pendulum.generate_table import generate_table as pendulum_table
from pointing.generate_table import generate_table as pointing_table
from somersault.generate_table import generate_table as somersault_table

import numpy as np
from bioptim import Shooting, OdeSolver


divergence_threshold = 10


class TableOCP:
    def __init__(self):
        self.cols = None

    def add(self, name):
        if not self.cols:
            self.cols = [TableOCP.OCP(name)]
        else:
            self.cols.append(TableOCP.OCP(name))

    def __getitem__(self, item_name):
        return self.cols[[col.name for col in self.cols].index(item_name)]

    def print(self):
        for col in self.cols:
            col.print()

    class OCP:
        def __init__(self, name):
            self.name = name
            self.solver = []

        def print(self):
            print(f"task = {self.name}")
            for solver in self.solver:
                solver.print()

        class Solver:
            def __init__(self, name):
                self.nx = -1
                self.nu = -1
                self.ns = -1
                self.name = name
                self.ode_solver = None
                self.n_iteration = -1
                self.cost = 0
                self.convergence_time = -1
                self.single_shoot_error_t = -1
                self.single_shoot_error_r = -1
                self.single_shoot_divergence_time = None

            def print(self):
                print(f"\t\tsolver = {self.name}")
                steps = f"internal steps = {self.ode_solver.steps}" if isinstance(self.ode_solver, OdeSolver.RK4) \
                    else f"polynomial degree = {self.ode_solver.polynomial_degree}"
                print(f"\t\t\tns = {type(self.ode_solver).__name__}, {steps}")
                print(f"\t\t\tns = {self.ns}")
                print(f"\t\t\tnx = {self.nx}")
                print(f"\t\t\tnu = {self.nu}")
                print(f"\t\t\titerations = {self.n_iteration}")
                print(f"\t\t\tcost = {self.cost}")
                print(f"\t\t\tconvergence_time (s) = {self.convergence_time}")
                print(f"\t\t\tsingle_shoot_error translation (mm) = {self.single_shoot_error_t}")
                print(f"\t\t\tsingle_shoot_error rotation (°) = {self.single_shoot_error_r}")
                print(f"\t\t\tsingle_shoot_time before divergence of {divergence_threshold}(°) = {self.single_shoot_divergence_time}")

            def compute_error_single_shooting(self, sol):
                sol_merged = sol.merge_phases()

                trans_idx = []
                rot_idx = []
                for i in sol.ocp.nlp[0].states["q"].mapping.to_second.map_idx:
                    if i is not None:
                        if sol.ocp.nlp[0].model.nameDof()[i].to_string()[-4:-1] == "Rot":
                            rot_idx += [i]
                        else:
                            trans_idx += [i]
                rot_idx = np.array(list(set(rot_idx)))
                trans_idx = np.array(list(set(trans_idx)))

                sol_int = sol.integrate(shooting_type=Shooting.SINGLE_CONTINUOUS, merge_phases=True, use_scipy_integrator=True)
                jumps = int((sol_merged.states["q"].shape[1] - 1) / (sol_int.states["q"].shape[1] - 1))
                if len(rot_idx) > 0:
                    error_r = sol_int.states["q"][rot_idx, :] - sol_merged.states["q"][rot_idx, ::jumps]
                    self.single_shoot_error_r = (np.sqrt(np.mean(error_r ** 2)) * 180 / np.pi)
                    for i in range(error_r.shape[1]):
                        if np.any(np.abs(error_r[:, i]) > divergence_threshold * np.pi / 180):
                            self.single_shoot_divergence_time = sol_int.phase_time[-1] / sol_int.ns[0] * i
                            break
                else:
                    self.single_shoot_error_r = "N.A."
                if len(trans_idx) > 0:
                    self.single_shoot_error_t = (
                        np.sqrt(
                            np.mean(
                                (sol_int.states["q"][trans_idx, :] - sol_merged.states["q"][trans_idx, ::jumps]) ** 2
                            )
                        )
                        / 1000
                    )
                else:
                    self.single_shoot_error_t = "N.A."


table = TableOCP()

table.add("gait")
table.add("jumper")
table.add("mhe")
table.add("pendulum")
table.add("pointing")
table.add("somersault")

gait_table(table["gait"])  # requires 16Gb of RAM and takes time to converge (~1h)
jumper_table(table["jumper"])
mhe_table(table["mhe"])
pendulum_table(table["pendulum"])
pointing_table(table["pointing"])
somersault_table(table["somersault"])

table.print()
