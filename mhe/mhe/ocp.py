import pickle

import biorbd_casadi as biorbd
import numpy as np
from casadi import MX, Function
from bioptim import (
    OptimalControlProgram,
    MovingHorizonEstimator,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsList,
    DynamicsFcn,
    BoundsList,
    QAndQDotBounds,
    InitialGuess,
    InterpolationType,
    Bounds
)


def muscle_forces(q: MX, qdot: MX, a: MX, controls: MX, model: biorbd.Model, use_excitations=False):
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
    use_excitations: bool
        True if excitations drive muscle dynamics False if activations driven
    Returns
    -------
    List of muscle forces
    """
    muscle_states = model.stateSet()
    for k in range(model.nbMuscles()):
        if use_excitations:
            muscle_states[k].setActivation(a[k])
            muscle_states[k].setExcitation(controls[k])
        else:
            muscle_states[k].setActivation(controls[k])
    return model.muscleForces(muscle_states, q, qdot).to_mx()


def get_reference_data(file_path):
    with open(file_path, "rb") as file:
        data = pickle.load(file)
    states = data["data"][0]
    controls = data["data"][1]
    return states["q"][:, 20:], states["qdot"][:, 20:], states["muscles"][:, 20:], controls["muscles"][:, 20:]


def muscle_force_func(biorbd_model: biorbd.Model, use_excitations=False):
    """
    Define Casadi function to use muscle_forces
    Parameters
    ----------
    biorbd_model: biorbd.Model
        biorbd model build with the bioMod
    use_excitations: bool
        True if excitation driven False if activation driven
    """
    q_mx = MX.sym("qMX", biorbd_model.nbQ(), 1)
    dq_mx = MX.sym("dq_mx", biorbd_model.nbQ(), 1)
    a_mx = MX.sym("a_mx", biorbd_model.nbMuscles(), 1)
    u_mx = MX.sym("u_mx", biorbd_model.nbMuscles(), 1)
    return Function(
        "MuscleForce",
        [q_mx, dq_mx, a_mx, u_mx],
        [muscle_forces(q_mx, dq_mx, a_mx, u_mx, biorbd_model, use_excitations=use_excitations)],
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
        noise = np.random.normal(0, abs(q_noise_lvl * (np.random.rand(1, q.shape[1]) * 0.01)))
        q_noise[i, :] = q[i, :] + noise
    return q_noise


def define_objective(
    target_q: np.array, iteration: int, rt_ratio: int, ns_mhe: int, use_noise=True, use_excitations=False
):
    """
    Define the objective function for the ocp
    Parameters
    ----------
    target_q: np.array
        State to track
    iteration: int
        Current iteration
    rt_ratio: int
        Value of the reference data ratio to send to the estimator
    ns_mhe: int
        Size of the windows
    use_noise: bool
        True if noisy reference data
    use_excitations: bool
        True if muscle excitations driven False if muscle activations driven
    Returns
    ---------
    The objective function
    """
    objectives = ObjectiveList()
    if use_noise is not True:
        if use_excitations:
            weight = {"track_state": 100000, "min_control": 1000, "min_dq": 10, "min_q": 10, "min_act": 1}
        else:
            weight = {"track_state": 10000, "min_control": 100, "min_dq": 100, "min_q": 100, "min_act": 10}
    else:
        if use_excitations:
            weight = {"track_state": 10000, "min_control": 10, "min_dq": 10, "min_q": 10, "min_act": 10}
        else:
            weight = {"track_state": 1000, "min_control": 10, "min_dq": 10, "min_q": 10, "min_act": 10}

    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", weight=weight["min_control"], multi_thread=False)
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=weight["min_q"], multi_thread=False)
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=weight["min_dq"], multi_thread=False)
    target_q = get_target(target_q, ns_mhe, rt_ratio, iteration)
    objectives.add(ObjectiveFcn.Lagrange.TRACK_STATE, key="q", weight=weight["track_state"], target=target_q, multi_thread=False)
    if use_excitations:
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="muscles", weight=weight["min_act"], multi_thread=False)
    return objectives


def get_target(q_ref, ns_mhe, rt_ratio, iteration):
    return q_ref[:, ::rt_ratio][:, iteration : (ns_mhe + 1 + iteration)]


def prepare_mhe(
        biorbd_model: biorbd.Model,
        final_time: float,
        n_shooting: int,
        x_ref: np.ndarray,
        rt_ratio: int,
        use_noise: bool,
        use_excitations=False,
    ):
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
    x_ref: np.ndarray
        The reference position
    rt_ratio

    Returns
    -------
    The blank OptimalControlProgram
    """

    q_ref = x_ref[: biorbd_model.nbQ(), 0: n_shooting * rt_ratio]
    # Add objective functions
    objective_functions = define_objective(q_ref, 0, rt_ratio, n_shooting, use_noise=use_noise, use_excitations=use_excitations)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.MUSCLE_DRIVEN, with_excitations=use_excitations)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))

    # Define control path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))
    x_bounds[0].min[: biorbd_model.nbQ(), 0] = q_ref[:, 0] - 0.1
    x_bounds[0].max[: biorbd_model.nbQ(), 0] = q_ref[:, 0] + 0.1
    if use_excitations:
        x_bounds[0].concatenate(
            Bounds([0] * biorbd_model.nbMuscles(), [1] * biorbd_model.nbMuscles())
        )

    u_bounds = BoundsList()
    u_bounds.add([0] * biorbd_model.nbMuscles(), [1] * biorbd_model.nbMuscles())

    x_init = InitialGuess(x_ref[:, : n_shooting + 1], interpolation=InterpolationType.EACH_FRAME)
    u_init = InitialGuess([0.2] * biorbd_model.nbMuscles(), interpolation=InterpolationType.CONSTANT)

    # ------------- #
    mhe = MovingHorizonEstimator(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
        objective_functions=objective_functions,
        use_sx=True,
        n_threads=8
    )

    solver_options = {
         "nlp_solver_tol_comp": 1e-8,
         "nlp_solver_tol_eq": 1e-8,
         "nlp_solver_tol_stat": 1e-8,
         "integrator_type": "IRK",
         "nlp_solver_type": "SQP",
         "sim_method_num_steps": 1,
         "print_level": 1,
         "nlp_solver_max_iter": 20,
    }

    return mhe, solver_options


def update_mhe(mhe, i, _, q_ref, ns_mhe, rt_ratio, final_time_index):
    target = get_target(q_ref, ns_mhe, rt_ratio, i)
    mhe.update_objectives_target(target, list_index=3)
    return i < final_time_index


def prepare_short_ocp(model_path: str, final_time: float, n_shooting: int, use_excitations=False):
    """
    Prepare to build a blank short ocp to use single shooting bioptim function
    Parameters
    ----------
    model_path: str
        path to bioMod
    final_time: float
        The time at the final node
    n_shooting: int
        The number of shooting points
    Returns
    -------
    The blank OptimalControlProgram
    """
    biorbd_model = biorbd.Model(model_path)

    # Add objective functions
    objective_functions = ObjectiveList()

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.MUSCLE_DRIVEN, with_excitations=use_excitations, expand=False)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))
    if use_excitations is True:
        x_bounds[0].concatenate(
            Bounds([0] * biorbd_model.nbMuscles(), [1] * biorbd_model.nbMuscles())
        )

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([0] * biorbd_model.nbMuscles(), [1] * biorbd_model.nbMuscles())

    x_init = [0] * (biorbd_model.nbQ() * 2 + biorbd_model.nbMuscles()) if use_excitations else [0] * biorbd_model.nbQ() * 2
    x_init = InitialGuess(x_init)
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
