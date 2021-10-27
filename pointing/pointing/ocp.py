import biorbd_casadi as biorbd
import numpy as np

from bioptim import (
    OptimalControlProgram,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsList,
    DynamicsFcn,
    BoundsList,
    QAndQDotBounds,
    InitialGuessList,
    Node,
    Bounds,
    Axis,
    OdeSolver,
)


def prepare_ocp(
    biorbd_model: biorbd.Model,
    final_time: float,
    n_shooting: int,
    use_sx: bool,
    weights: np.ndarray,
    use_excitations=False,
    use_collocation=False,
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    biorbd_model: str
        The path to the bioMod
    final_time: float
        The time at the final node
    n_shooting: int
        The number of shooting points
    use_sx: bool
        If SX should be used or not
    weights: float
        The weight applied to the SUPERIMPOSE_MARKERS final objective function. The bigger this number is, the greater
        the model will try to reach the marker. This is in relation with the other objective functions

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    if not use_sx and use_collocation:
        raise RuntimeError("Acados solver cannot be used with colocations.")
    ode_solver = OdeSolver.COLLOCATION() if use_collocation else OdeSolver.RK4()

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=weights[0], multi_thread=False)
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", node=Node.ALL, weight=weights[1], multi_thread=False
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", node=Node.ALL, weight=weights[1], multi_thread=False
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", weight=weights[2], multi_thread=False
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS,
        first_marker=0,
        second_marker=1,
        axes=[Axis.X, Axis.Y],
        weight=weights[3],
    )
    # if not use_sx:
    if use_excitations:
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="muscles", weight=weights[4], multi_thread=False
        )

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.MUSCLE_DRIVEN, with_torque=True, with_excitations=use_excitations, expand=False)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))

    # Force initial position
    if use_sx:
        x_bounds[0][:, 0] = [1.24, 1.55, 0, 0]
    else:
        x_bounds[0][:, 0] = [1.0, 1.3, 0, 0]

    if use_excitations:
        x_bounds[0].concatenate(Bounds([0] * biorbd_model.nbMuscles(), [1] * biorbd_model.nbMuscles()))
    # Initial guess
    x_init = InitialGuessList()
    init_state = [1.57] * biorbd_model.nbQ() + [0] * biorbd_model.nbQdot()
    if use_excitations:
        init_state = init_state + [0.2] * biorbd_model.nbMuscles()
    x_init.add(init_state)

    # Define control path constraint
    muscle_min, muscle_max, muscle_init = 0, 1, 0.1
    tau_min, tau_max, tau_init = -20, 20, 0
    u_bounds = BoundsList()
    u_bounds.add(
        [tau_min] * biorbd_model.nbGeneralizedTorque() + [muscle_min] * biorbd_model.nbMuscleTotal(),
        [tau_max] * biorbd_model.nbGeneralizedTorque() + [muscle_max] * biorbd_model.nbMuscleTotal(),
    )
    # u_bounds[0][:, 0] = [0] * biorbd_model.nbGeneralizedTorque() + [0] * biorbd_model.nbMuscleTotal()
    u_init = InitialGuessList()
    u_init.add([tau_init] * biorbd_model.nbGeneralizedTorque() + [muscle_init] * biorbd_model.nbMuscleTotal())
    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        n_threads=8,
        use_sx=use_sx,
        ode_solver=ode_solver,
    )
