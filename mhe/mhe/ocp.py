import biorbd
import numpy as np
from casadi import MX, Function
from bioptim import (
    OptimalControlProgram,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsList,
    DynamicsFcn,
    BoundsList,
    QAndQDotBounds,
    InitialGuess,
)


def muscle_forces(q: MX, qdot: MX, a: MX, controls: MX, model: biorbd.Model, use_activation=True):
    """
    Compute muscle force
    Parameters
    ----------
    q: MX
        Symbolic value of joint angle
    qdot: MX
        Symbolic value of joint velocity
    a: MX
        Symbolic activation
    controls: int
        Symbolic value of activations
    model: biorbd.Model
        biorbd model build with the bioMod
    use_activation: bool
        True if activation drive False if excitation driven
    Returns
    -------
    List of muscle forces
    """
    muscle_states = biorbd.VecBiorbdMuscleState(model.nbMuscles())
    for k in range(model.nbMuscles()):
        if use_activation:
            muscle_states[k].setActivation(controls[k])
        else:
            muscle_states[k].setActivation(a[k])
            muscle_states[k].setExcitation(controls[k])
    return model.muscleForces(muscle_states, q, qdot).to_mx()


def force_func(biorbd_model: biorbd.Model, use_activation=True):
    """
    Define Casadi function to use muscle_forces
    Parameters
    ----------
    biorbd_model: biorbd.Model
        biorbd model build with the bioMod
    use_activation: bool
        True if activation drive False if excitation driven
    """
    q_mx = MX.sym("qMX", biorbd_model.nbQ(), 1)
    dq_mx = MX.sym("dq_mx", biorbd_model.nbQ(), 1)
    a_mx = MX.sym("a_mx", biorbd_model.nbMuscles(), 1)
    u_mx = MX.sym("u_mx", biorbd_model.nbMuscles(), 1)
    return Function(
        "MuscleForce",
        [q_mx, dq_mx, a_mx, u_mx],
        [muscle_forces(q_mx, dq_mx, a_mx, u_mx, biorbd_model, use_activation=use_activation)],
        ["qMX", "dq_mx", "a_mx", "u_mx"],
        ["Force"],
    ).expand()


def generate_noise(biorbd_model, q: np.array, q_noise_lvl: float):
    """
    Generate random Centered Gaussian noise apply on joint angles
    Parameters
    ----------
    biorbd_model: biorbd.Model
        biorbd model build with the bioMod
    q: np.array
        Array of reference joint angles
    q_noise_lvl: float
        Standard deviation value in percent
    Returns
    ---------
    Array of noisy joint angles
    """
    n_q = biorbd_model.nbQ()
    q_noise = np.ndarray((n_q, q.shape[1]))
    for i in range(n_q):
        noise = np.random.normal(0, abs(q_noise_lvl * q[i, :] / 100))
        q_noise[i, :] = q[i, :] + noise
    return q_noise


def warm_start_mhe(sol):
    """
    Ensures the problems continuity
    Parameters
    ----------
    sol: sol
        the solutions of the previous problem
    Returns
    ---------
    Initial states and controls for next problem (x0, u0)
    States and controls to save as solution (x_out, u_out)
    """
    x = sol.states["all"]
    u = sol.controls["all"]

    x0 = np.hstack((x[:, 1:], np.tile(x[:, [-1]], 1)))  # discard oldest estimate of the window, duplicates youngest
    u0 = u[:, :-1]
    x_out = x[:, 0]
    u_out = u[:, 0]
    return x0, u0, x_out, u_out


def define_objective(
    q: np.array, iteration: int, rt_ratio: int, ns_mhe: int, biorbd_model: biorbd.Model, use_noise=True
):
    """
    Define the objective function for the ocp
    Parameters
    ----------
    q: np.array
        State to track
    iteration: int
        Current iteration
    rt_ratio: int
        Value of the reference data ratio to send to the estimator
    ns_mhe: int
        Size of the windows
    biorbd_model: biorbd.Model
        biorbd model build with the bioMod
    use_noise: bool
        True if noisy reference data
    Returns
    ---------
    The objective function
    """
    objectives = ObjectiveList()
    if use_noise is not True:
        weight = {"track_state": 1000000, "min_act": 1000, "min_dq": 10, "min_q": 10}
    else:
        weight = {"track_state": 1000, "min_act": 100, "min_dq": 100, "min_q": 10}
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=weight["min_act"])
    objectives.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        weight=weight["min_dq"],
        index=np.array(range(biorbd_model.nbQ(), biorbd_model.nbQ() * 2)),
    )
    objectives.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        weight=weight["min_q"],
        index=np.array(range(biorbd_model.nbQ())),
    )
    q = q[:, ::rt_ratio]
    objectives.add(
        ObjectiveFcn.Lagrange.TRACK_STATE,
        weight=weight["track_state"],
        target=q[:, iteration : (ns_mhe + 1 + iteration)],
        index=range(biorbd_model.nbQ()),
    )
    return objectives


def prepare_ocp(biorbd_model: biorbd.Model, final_time: float, n_shooting: int):
    """
    Prepare to build a blank ocp witch will be update several times
    Parameters
    ----------
    biorbd_model: biorbd.Model
        biorbd model build with the bioMod
    final_time: float
        The time at the final node
    n_shooting: int
        The number of shooting points
    Returns
    -------
    The blank OptimalControlProgram
    """
    # Add objective functions
    objective_functions = ObjectiveList()

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.MUSCLE_ACTIVATIONS_DRIVEN)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([0] * biorbd_model.nbMuscles(), [1] * biorbd_model.nbMuscles())

    x_init = InitialGuess([0] * biorbd_model.nbQ() * 2)
    u_init = InitialGuess([0] * biorbd_model.nbMuscles())

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
        use_sx=True,
    )


def prepare_short_ocp(biorbd_model: biorbd.Model, final_time: float, n_shooting: int):
    """
    Prepare to build a blank short ocp to use single shooting bioptim function
    Parameters
    ----------
    biorbd_model: biorbd.Model
        biorbd model build with the bioMod
    final_time: float
        The time at the final node
    n_shooting: int
        The number of shooting points
    Returns
    -------
    The blank OptimalControlProgram
    """
    # Add objective functions
    objective_functions = ObjectiveList()

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.MUSCLE_ACTIVATIONS_DRIVEN)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([0] * biorbd_model.nbMuscles(), [1] * biorbd_model.nbMuscles())

    x_init = InitialGuess([0] * biorbd_model.nbQ() * 2)
    u_init = InitialGuess([0] * biorbd_model.nbMuscles())

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
        use_sx=True,
    )
