import casadi as cas
import biorbd
import numpy as np
from bioptim import (
    OptimalControlProgram,
    DynamicsList,
    ObjectiveList,
    ObjectiveFcn,
    Bounds,
    BoundsList,
    InitialGuessList,
    InterpolationType,
    Problem,
    DynamicsFunctions,
)


def custom_dynamic(states, controls, parameters, nlp):
    q, qdot, tau = DynamicsFunctions.dispatch_q_qdot_tau_data(states, controls, nlp)

    force_vector = cas.MX.zeros(6)
    force_vector[5] = -200 * q[0]

    f_ext = biorbd.VecBiorbdSpatialVector()
    f_ext.append(biorbd.SpatialVector(force_vector))
    qddot = nlp.model.ForwardDynamics(q, qdot, tau, f_ext).to_mx()

    dxdt = cas.vertcat(qdot, qddot)

    return dxdt


def custom_configure(ocp, nlp):
    Problem.configure_q_qdot(nlp, as_states=True, as_controls=False)
    Problem.configure_tau(nlp, as_states=False, as_controls=True)
    Problem.configure_dynamics_function(ocp, nlp, custom_dynamic)


def prepare_ocp(biorbd_model_path: str, use_sx: bool = False) -> OptimalControlProgram:
    """
    Prepare the ocp
    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod
    use_sx: bool
        If the project should be build in mx [False] or sx [True]
    Returns
    -------
    The OptimalControlProgram ready to be solved
    """
    # Model path
    biorbd_model = (
        biorbd.Model(biorbd_model_path),
        biorbd.Model(biorbd_model_path),
    )

    # Problem parameters
    number_shooting_points = (
        50,
        50,
    )
    final_time = (
        5,
        5,
    )
    tau_min, tau_max, tau_init = -500, 500, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Lagrange.TRACK_STATE,
        weight=1,
        index=0,
        target=np.ones((1, number_shooting_points[0] + 1)) * -0.5,
        phase=1,
    )
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=1e-6, phase=1)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(custom_configure, dynamic_function=custom_dynamic)
    dynamics.add(custom_configure, dynamic_function=custom_dynamic)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(
        bounds=Bounds(
            np.array([-10, -4 * np.pi, -1000, -1000]),
            np.array([10, 4 * np.pi, 1000, 1000]),
            interpolation=InterpolationType.CONSTANT,
        )
    )
    x_bounds.add(
        bounds=Bounds(
            np.array([-10, -4 * np.pi, -1000, -1000]),
            np.array([10, 4 * np.pi, 1000, 1000]),
            interpolation=InterpolationType.CONSTANT,
        )
    )

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add(bounds=Bounds([0, 0], [0, 0]))
    u_bounds.add(bounds=Bounds([tau_min, 0], [tau_max, 0]))

    # Initial guess
    x_init = InitialGuessList()
    x_init.add(np.random.random(biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))
    x_init.add(np.random.random(biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))

    u_init = InitialGuessList()
    u_init.add([0, 0])
    u_init.add([0, 0])

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        number_shooting_points,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        n_threads=8,
        use_sx=use_sx,
    )
