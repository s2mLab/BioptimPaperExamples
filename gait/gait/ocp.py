import numpy as np
from casadi import  vertcat, MX, sum1
import biorbd_casadi
from bioptim import (
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    BoundsList,
    QAndQDotBounds,
    InitialGuessList,
    ObjectiveList,
    ObjectiveFcn,
    InterpolationType,
    Node,
    ConstraintList,
    ConstraintFcn,
    PhaseTransitionList,
    PhaseTransitionFcn,
    PenaltyNode,
    BiorbdInterface,
    Axis,
)


def get_contact_index(pn, tag):
    force_names = [s.to_string() for s in pn.nlp.model.contactNames()]
    return [i for i, t in enumerate([s[-1] == tag for s in force_names]) if t]


# --- force nul at last point ---
def force_contact(pn: PenaltyNode, index_contact: list) -> MX:
    """
    Adds the constraint that the force at the specific contact point should be null
    at the last phase point.
    All contact forces can be set at 0 at the last node by using 'all' at contact_name.

    Parameters
    ----------
    pn: PenaltyNode
        The penalty node elements

    Returns
    -------
    The value that should be constrained in the MX format

    """

    states = vertcat(pn.nlp.states["q"].mx, pn.nlp.states["qdot"].mx)
    controls = vertcat(pn.nlp.controls["tau"].mx, pn.nlp.controls["muscles"].mx)
    force = pn.nlp.contact_forces_func(states, controls, pn.nlp.parameters.mx)
    return BiorbdInterface.mx_to_cx("grf", force, pn.nlp.states["q"], pn.nlp.states["qdot"], pn.nlp.controls["tau"], pn.nlp.controls["muscles"])[index_contact]


# --- track grf ---
def track_sum_contact_forces(pn: PenaltyNode) -> MX:
    """
    Adds the objective that the mismatch between the
    sum of the contact forces and the reference ground reaction forces should be minimized.

    Parameters
    ----------
    pn: PenaltyNode
        The penalty node elements

    Returns
    -------
    The cost that should be minimize in the MX format.
    """
    states = vertcat(pn.nlp.states["q"].mx, pn.nlp.states["qdot"].mx)
    controls = vertcat(pn.nlp.controls["tau"].mx, pn.nlp.controls["muscles"].mx)
    force_tp = pn.nlp.contact_forces_func(states, controls, pn.nlp.parameters.mx)

    force = vertcat(sum1(force_tp[get_contact_index(pn, "X"), :]),
                    sum1(force_tp[get_contact_index(pn, "Y"), :]),
                    sum1(force_tp[get_contact_index(pn, "Z"), :]))
    return BiorbdInterface.mx_to_cx("grf", force, pn.nlp.states["q"], pn.nlp.states["qdot"], pn.nlp.controls["tau"], pn.nlp.controls["muscles"])


# --- track grf ---
def track_sum_contact_moments(pn: PenaltyNode, marker_foot: list) -> MX:
    """
    Adds the objective that the mismatch between the
    sum of the contact forces and the reference ground reaction forces should be minimized.

    Parameters
    ----------
    pn: PenaltyNode
        The penalty node elements

    Returns
    -------
    The cost that should be minimize in the MX format.
    """
    states = vertcat(pn.nlp.states["q"].mx, pn.nlp.states["qdot"].mx)
    controls = vertcat(pn.nlp.controls["tau"].mx, pn.nlp.controls["muscles"].mx)
    force_tp = pn.nlp.contact_forces_func(states, controls, pn.nlp.parameters.mx)
    force_x = force_tp[get_contact_index(pn, "X"), :]
    force_y = force_tp[get_contact_index(pn, "Y"), :]
    force_z = force_tp[get_contact_index(pn, "Z"), :]

    q = pn.nlp.states["q"].mx
    markers = biorbd_casadi.to_casadi_func("markers", pn.nlp.model.markers, pn.nlp.states["q"].mx)(q)
    cop = vertcat(sum1(vertcat(*[-markers[0, m] * force_z for m in marker_foot])) / sum1(force_z),
                  sum1(vertcat(*[-markers[1, m] * force_z for m in marker_foot])) / sum1(force_z),
                  0)

    # --- moments --- #
    markers_cop = markers - cop
    moments = vertcat(sum1(vertcat(*[markers_cop[1, m] * force_z for m in marker_foot])),  # y*fz
                      sum1(vertcat(*[-markers_cop[0, m] * force_z for m in marker_foot])),  # -x*fz
                      sum1(vertcat(*[markers_cop[0, m] * force_y for m in marker_foot])) - sum1(vertcat(*[markers_cop[1, m] * force_x for m in marker_foot])))  # x*fy - y*fx
    return BiorbdInterface.mx_to_cx("moments", moments, pn.nlp.states["q"], pn.nlp.states["qdot"],
                                    pn.nlp.controls["tau"], pn.nlp.controls["muscles"])


def prepare_ocp(
    biorbd_model: tuple,
    final_time: list,
    nb_shooting: list,
    markers_ref: list,
    grf_ref: list,
    q_ref: list,
    qdot_ref: list,
    M_ref: list,
    nb_threads: int,
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    biorbd_model: tuple
        Tuple of bioMod (1 bioMod for each phase)
    final_time: list
        List of the time at the final node.
        The length of the list corresponds to the phase number
    nb_shooting: list
        List of the number of shooting points
    markers_ref: list
        List of the array of markers trajectories to track
    grf_ref: list
        List of the array of ground reaction forces to track
    q_ref: list
        List of the array of joint trajectories.
        Those trajectories were computed using Kalman filter
        They are used as initial guess
    qdot_ref: list
        List of the array of joint velocities.
        Those velocities were computed using Kalman filter
        They are used as initial guess
    M_ref: list
        List of the array of ground reaction moments to track
    nb_threads:int
        The number of threads used

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    # Problem parameters
    nb_phases = len(biorbd_model)
    nb_q = biorbd_model[0].nbQ()
    nb_qdot = biorbd_model[0].nbQdot()
    nb_tau = biorbd_model[0].nbGeneralizedTorque()
    nb_mus = biorbd_model[0].nbMuscleTotal()

    min_bound, max_bound = 0, np.inf
    torque_min, torque_max, torque_init = -1000, 1000, 0
    activation_min, activation_max, activation_init = 1e-3, 1.0, 0.1

    # Add objective functions
    markers_pelvis = [0, 1, 2, 3]
    markers_anat = [4, 9, 10, 11, 12, 17, 18]
    markers_tissus = [5, 6, 7, 8, 13, 14, 15, 16]
    markers_foot = [19, 20, 21, 22, 23, 24, 25]
    objective_functions = ObjectiveList()
    for p in range(nb_phases):
        # objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, key="q", node=Node.ALL, target=q_ref[p], quadratic=True, phase=p)
        objective_functions.add(
            ObjectiveFcn.Lagrange.TRACK_MARKERS,
            node=Node.ALL,
            weight=1000,
            marker_index=markers_anat,
            target=markers_ref[p][:, markers_anat, :],
            quadratic=True,
            phase=p,
        )
        objective_functions.add(
            ObjectiveFcn.Lagrange.TRACK_MARKERS,
            node=Node.ALL,
            weight=100000,
            marker_index=markers_pelvis,
            target=markers_ref[p][:, markers_pelvis, :],
            quadratic=True,
            phase=p,
        )
        objective_functions.add(
            ObjectiveFcn.Lagrange.TRACK_MARKERS,
            node=Node.ALL,
            weight=10000,
            marker_index=markers_foot,
            target=markers_ref[p][:, markers_foot, :],
            quadratic=True,
            phase=p,
        )
        objective_functions.add(
            ObjectiveFcn.Lagrange.TRACK_MARKERS,
            node=Node.ALL,
            weight=100,
            marker_index=markers_tissus,
            target=markers_ref[p][:, markers_tissus, :],
            quadratic=True,
            phase=p,
        )
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.001, index=10, quadratic=True, phase=p)
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, index=(6, 7, 8, 9, 11), phase=p, quadratic=True,
        )
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", weight=10, phase=p, quadratic=True,)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=0.1, quadratic=True, phase=p)

    # --- track contact forces for the stance phase ---
    for p in range(nb_phases - 1):
        objective_functions.add(
            track_sum_contact_forces,  # track contact forces
            custom_type=ObjectiveFcn.Lagrange,
            target=grf_ref[p],
            node=Node.ALL,
            weight=0.01,
            quadratic=True,
            phase=p,
        )

    # for p in range(1, nb_phases - 1):
    #     objective_functions.add(
    #         track_sum_contact_moments,
    #         target=M_ref[p],
    #         custom_type=ObjectiveFcn.Lagrange,
    #         node=Node.ALL,
    #         weight=0.01,
    #         quadratic=True,
    #         phase=p,
    #         marker_foot=[26, 27, 28, 29]
    #     )

    # Dynamics
    dynamics = DynamicsList()
    for p in range(nb_phases - 1):
        dynamics.add(DynamicsFcn.MUSCLE_DRIVEN, phase=p, with_contact=True, with_torque=True, expand=False)
    dynamics.add(DynamicsFcn.MUSCLE_DRIVEN, phase=3, with_torque=True, expand=False)

    # Constraints
    constraints = ConstraintList()
    constraints.add(  # null speed for the first phase --> non sliding contact point
        ConstraintFcn.TRACK_MARKERS_VELOCITY,
        node=Node.START,
        marker_index=26,
        phase=0,
    )
    constraints.add(  # on the ground z=0
        ConstraintFcn.TRACK_MARKERS,
        node=Node.START,
        marker_index=26,
        axes=Axis.Z,
        phase=0,
    )

    # --- phase flatfoot ---
    constraints.add(  # on the ground z=0
        ConstraintFcn.TRACK_MARKERS,
        node=Node.START,
        marker_index=[27, 28],
        axes=Axis.Z,
        phase=1,
    )

    constraints.add(  # positive vertical forces
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.ALL,
        contact_index=(1, 2, 5),
        phase=1,
    )
    constraints.add(  # non slipping x m5
        ConstraintFcn.NON_SLIPPING,
        node=Node.ALL,
        normal_component_idx=5,
        tangential_component_idx=3,
        static_friction_coefficient=0.2,
        phase=1,
    )
    constraints.add(  # non slipping y m5
        ConstraintFcn.NON_SLIPPING,
        node=Node.ALL,
        normal_component_idx=5,
        tangential_component_idx=4,
        static_friction_coefficient=0.2,
        phase=1,
    )
    constraints.add(  # non slipping x heel
        ConstraintFcn.NON_SLIPPING,
        node=Node.ALL,
        normal_component_idx=1,
        tangential_component_idx=0,
        static_friction_coefficient=0.2,
        phase=1,
    )

    constraints.add(  # forces heel at zeros at the end of the phase
        force_contact,
        node=Node.PENULTIMATE,
        index_contact=[i for i, name in enumerate(biorbd_model[1].contactNames()) if "Heel_r" in name.to_string()],
        phase=1,
    )

    # --- phase forefoot ---
    constraints.add(  # positive vertical forces
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.ALL,
        contact_index=(2, 4, 5),
        phase=2,
    )
    constraints.add( # non slipping x m1
        ConstraintFcn.NON_SLIPPING,
        node=Node.ALL,
        normal_component_idx=2,
        tangential_component_idx=0,
        static_friction_coefficient=0.2,
        phase=2,
    )
    constraints.add(  # non slipping y m1
        ConstraintFcn.NON_SLIPPING,
        node=Node.ALL,
        normal_component_idx=2,
        tangential_component_idx=1,
        static_friction_coefficient=0.2,
        phase=2,
    )
    constraints.add(  # non slipping x m5
        ConstraintFcn.NON_SLIPPING,
        node=Node.ALL,
        normal_component_idx=4,
        tangential_component_idx=3,
        static_friction_coefficient=0.2,
        phase=2,
    )

    # Phase Transitions
    phase_transitions = PhaseTransitionList()
    phase_transitions.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=0)
    phase_transitions.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=1)

    # Path constraint
    x_bounds = BoundsList()
    u_bounds = BoundsList()
    for p in range(nb_phases):
        x_bounds.add(bounds=QAndQDotBounds(biorbd_model[p]))
        u_bounds.add(
            [torque_min] * nb_tau + [activation_min] * nb_mus,
            [torque_max] * nb_tau + [activation_max] * nb_mus,
        )

    # Initial guess
    x_init = InitialGuessList()
    u_init = InitialGuessList()
    n_shoot = 0
    for p in range(nb_phases):
        init_x = np.zeros((nb_q + nb_qdot, nb_shooting[p] + 1))
        init_x[:nb_q, :] = q_ref[p]
        init_x[nb_q : nb_q + nb_qdot, :] = qdot_ref[p]
        x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)

        init_u = [torque_init] * nb_tau + [activation_init] * nb_mus
        u_init.add(init_u)
        n_shoot += nb_shooting[p]

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        nb_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        constraints,
        phase_transitions=phase_transitions,
        n_threads=nb_threads,
    )


def get_phase_time_shooting_numbers(data, dt):
    phase_time = data.c3d_data.get_time()
    number_shooting_points = []
    for time in phase_time:
        number_shooting_points.append(int(time / dt) - 1)
    return phase_time, number_shooting_points


def get_experimental_data(data, number_shooting_points):
    q_ref = data.dispatch_data(data=data.q, nb_shooting=number_shooting_points)
    qdot_ref = data.dispatch_data(data=data.qdot, nb_shooting=number_shooting_points)
    markers_ref = data.dispatch_data(data=data.c3d_data.trajectories, nb_shooting=number_shooting_points)
    grf_ref = data.dispatch_data(data=data.c3d_data.forces, nb_shooting=number_shooting_points)
    moments_ref = data.dispatch_data(data=data.c3d_data.moments, nb_shooting=number_shooting_points)
    cop_ref = data.dispatch_data(data=data.c3d_data.cop, nb_shooting=number_shooting_points)
    return q_ref, qdot_ref, markers_ref, grf_ref, moments_ref, cop_ref
