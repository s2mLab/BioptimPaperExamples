from time import perf_counter

from bioptim import ControlType

from JumperOcp import JumperOcp, Jumper, OdeSolver

if __name__ == "__main__":
    root_path_model = "/".join(__file__.split("/")[:-1])
    jumper_model = Jumper(root_path_model + "/models/")
    ode_solver = OdeSolver.RK4()
    jumper = JumperOcp(jumper=jumper_model, control_type=ControlType.CONSTANT, ode_solver=ode_solver)

    tic = perf_counter()

    # from bioptim import Solution, InitialGuess
    # sol = Solution(jumper.ocp, [jumper.x_init[0], jumper.u_init[0]])
    # sol.integrate().animate(show_meshes=False)
    # #

    sol = jumper.solve(limit_memory_max_iter=0, exact_max_iter=1000)
    print(f"Time to solve : {perf_counter() - tic}sec")
    #
    sol.print()
    sol.animate(show_meshes=False)
