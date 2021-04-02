import biorbd
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
)


def prepare_ocp(
    biorbd_model: biorbd.Model,
    final_time: float,
    n_shooting: int,
    use_sx: bool,
    weights: np.ndarray,
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

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=weights[0])
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, weight=weights[1])
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=weights[2])
    objective_functions.add(
        ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS, first_marker_idx=0, second_marker_idx=1, weight=weights[3]
    )

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))

    # Force initial position
    if use_sx:
        x_bounds[0][:, 0] = [1.24, 1.55, 0, 0]
    else:
        x_bounds[0][:, 0] = [1.0, 1.3, 0, 0]
    # Initial guess
    x_init = InitialGuessList()
    init_state = [1.57] * biorbd_model.nbQ() + [0] * biorbd_model.nbQdot()
    x_init.add(init_state)

    # Define control path constraint
    muscle_min, muscle_max, muscle_init = 0, 1, 0.5
    tau_min, tau_max, tau_init = -10, 10, 0
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
    )
