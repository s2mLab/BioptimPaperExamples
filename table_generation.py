from gait.generate_table import generate_table as gait_table
from jumper.generate_table import generate_table as jumper_table
from mhe.generate_table import generate_table as mhe_table
from pendulum.generate_table import generate_table as pendulum_table
from pointing.generate_table import generate_table as pointing_table
from somersault.generate_table import generate_table as somersault_table

import numpy as np
from bioptim import Shooting


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
            self.nx = -1
            self.nu = -1
            self.ns = -1
            self.solver = []

        def print(self):
            print(f"task = {self.name}")
            print(f"\tns = {self.ns}")
            print(f"\tnx = {self.nx}")
            print(f"\tnu = {self.nu}")
            for solver in self.solver:
                solver.print()

        class Solver:
            def __init__(self, name):
                self.name = name
                self.n_iteration = -1
                self.cost = 0
                self.convergence_time = -1
                self.single_shoot_error_t = -1
                self.single_shoot_error_r = -1

            def print(self):
                print(f"\t\tsolver = {self.name}")
                print(f"\t\t\titerations = {self.n_iteration}")
                print(f"\t\t\tcost = {self.cost}")
                print(f"\t\t\tconvergence_time (s) = {self.convergence_time}")
                print(f"\t\t\tsingle_shoot_error translation (mm) = {self.single_shoot_error_t}")
                print(f"\t\t\tsingle_shoot_error rotation (°) = {self.single_shoot_error_r}")

            def compute_error_single_shooting(self, sol, duration, use_final_time=False):
                sol_merged = sol.merge_phases()

                if sol_merged.phase_time[-1] < duration and not use_final_time:
                    raise ValueError(
                        f"Single shooting integration duration must be smaller than "
                        f"ocp duration: {sol_merged.phase_time[-1]} s. "
                        f"You can set use_final_time=True if you want to use the final time for the "
                        f"Single shooting integration duration"
                    )

                trans_idx = []
                rot_idx = []
                for i in sol.ocp.nlp[0].mapping["q"].to_second.map_idx:
                    if i is not None:
                        if sol.ocp.nlp[0].model.nameDof()[i].to_string()[-4:-1] == "Rot":
                            rot_idx += [i]
                        else:
                            trans_idx += [i]
                rot_idx = np.array(list(set(rot_idx)))
                trans_idx = np.array(list(set(trans_idx)))

                sol_int = sol.integrate(shooting_type=Shooting.SINGLE_CONTINUOUS, merge_phases=False, keepdims=True)
                sol_int = sol_int.merge_phases()
                if use_final_time:
                    sn_1s = -1
                else:
                    sn_1s = int(sol_int.ns[0] / sol_int.phase_time[-1] * duration)  # shooting node at {duration} second
                if len(rot_idx) > 0:
                    self.single_shoot_error_r = (
                        np.sqrt(
                            np.mean((sol_int.states["q"][rot_idx, sn_1s] - sol_merged.states["q"][rot_idx, sn_1s]) ** 2)
                        )
                        * 180
                        / np.pi
                    )
                else:
                    self.single_shoot_error_r = "N.A."
                if len(trans_idx) > 0:
                    self.single_shoot_error_t = (
                        np.sqrt(
                            np.mean(
                                (sol_int.states["q"][trans_idx, sn_1s] - sol_merged.states["q"][trans_idx, sn_1s]) ** 2
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

gait_table(table["gait"])
jumper_table(table["jumper"])
mhe_table(table["mhe"])
pendulum_table(table["pendulum"])
pointing_table(table["pointing"])
somersault_table(table["somersault"])

table.print()
