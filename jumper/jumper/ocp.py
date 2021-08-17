import numpy as np
import biorbd_casadi as biorbd
from bioptim import (
    PenaltyNode,
    Node,
    ConstraintList,
    ConstraintFcn,
    ObjectiveFcn,
    ObjectiveList,
    DynamicsList,
    DynamicsFcn,
    BoundsList,
    QAndQDotBounds,
    PhaseTransitionList,
    PhaseTransitionFcn,
    BiMappingList,
    InitialGuess,
    InterpolationType,
    OptimalControlProgram,
    InitialGuessList,
    BiorbdInterface,
)


def com_dot_z(nodes: PenaltyNode):
    z = nodes.nlp.model.CoMdot(nodes.nlp.states["q"].mx, nodes.nlp.states["qdot"].mx).to_mx()[2, 0]
    return BiorbdInterface.mx_to_cx("com_dot", z, nodes.nlp.states["q"], nodes.nlp.states["qdot"])


def toe_on_floor(nodes: PenaltyNode, floor: float):
    toe_marker_z = nodes.nlp.model.marker(nodes.nlp.states["q"].mx, 2).to_mx()[2, 0]
    return BiorbdInterface.mx_to_cx("toe_marker_z", toe_marker_z + floor, nodes.nlp.states["q"])


def heel_on_floor(nodes: PenaltyNode, floor: float):
    toe_marker_z = nodes.nlp.model.marker(nodes.nlp.states["q"].mx, 3).to_mx()[2, 0]
    return BiorbdInterface.mx_to_cx("toe_marker_z", toe_marker_z + floor, nodes.nlp.states["q"])


class Jumper5Phases:
    floor = 0.779  # floor = -0.77865438

    def __init__(self, model_paths, n_shoot, time_min, phase_time, time_max, initial_pose, n_thread=1):
        self.models = []
        self._load_models(model_paths)

        # Element for the optimization
        self.n_phases = 5
        self.n_shoot = n_shoot
        self.time_min = time_min
        self.phase_time = phase_time
        self.time_max = time_max
        self.takeoff = 1  # The index of takeoff phase
        self.flat_foot_phases = 0, 4  # The indices of flat foot phases
        self.toe_only_phases = 1, 3  # The indices of toe only phases

        # Elements from the model
        self.initial_states = []
        self._set_initial_states(initial_pose, [0, 0, 0, 0, 0, 0, 0])
        self.heel_and_toe_idx = (1, 2, 4, 5)  # Contacts indices of heel and toe in bioMod 2 contacts
        self.toe_idx = (1, 3)  # Contacts indices of toe in bioMod 1 contact
        self.n_q, self.n_qdot, self.n_tau = -1, -1, -1
        self.mapping_list = BiMappingList()
        self._set_dimensions_and_mapping()

        # Prepare the optimal control program
        self.dynamics = DynamicsList()
        self._set_dynamics()

        self.constraints = ConstraintList()
        self.tau_min = 20
        self._set_constraints()

        self.objective_functions = ObjectiveList()
        self._set_objective_functions()

        self.x_bounds = BoundsList()
        self.u_bounds = BoundsList()
        self._set_boundary_conditions()

        self.phase_transitions = PhaseTransitionList()
        self._set_phase_transitions()

        self.x_init = InitialGuessList()
        self.u_init = InitialGuessList()
        self._set_initial_guesses()

        self.ocp = OptimalControlProgram(
            self.models,
            self.dynamics,
            self.n_shoot,
            self.phase_time,
            x_init=self.x_init,
            x_bounds=self.x_bounds,
            u_init=self.u_init,
            u_bounds=self.u_bounds,
            objective_functions=self.objective_functions,
            constraints=self.constraints,
            variable_mappings=self.mapping_list,
            phase_transitions=self.phase_transitions,
            n_threads=n_thread,
        )

    def _load_models(self, model_paths):
        self.models = [biorbd.Model(elt) for elt in model_paths]

    def _set_initial_states(self, initial_pose, initial_velocity):
        self.initial_states = np.array([list(initial_pose) + initial_velocity]).T

    def _set_dimensions_and_mapping(self):
        self.mapping_list.add("q", [0, 1, 2, 3, 3, 4, 5, 6, 4, 5, 6], [0, 1, 2, 3, 5, 6, 7])
        self.mapping_list.add("qdot", [0, 1, 2, 3, 3, 4, 5, 6, 4, 5, 6], [0, 1, 2, 3, 5, 6, 7])
        self.mapping_list.add("tau", [None, None, None, 0, 0, 1, 2, 3, 1, 2, 3], [3, 5, 6, 7])
        self.n_q = len(self.mapping_list["q"].to_first)
        self.n_qdot = self.n_q
        self.n_tau = len(self.mapping_list["tau"].to_first)

    def _set_dynamics(self):
        self.dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True, expand=False)  # Flat foot
        self.dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True, expand=False)  # Toe only
        self.dynamics.add(DynamicsFcn.TORQUE_DRIVEN)  # Aerial phase
        self.dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True, expand=False)  # Toe only
        self.dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True, expand=False)  # Flat foot

    def _set_constraints(self):
        # Torque constrained to torqueMax
        for i in range(self.n_phases):
            self.constraints.add(ConstraintFcn.TORQUE_MAX_FROM_Q_AND_QDOT, phase=i, node=Node.ALL_SHOOTING, min_torque=self.tau_min)

        # Positivity of CoM_dot on z axis prior the take-off
        self.constraints.add(com_dot_z, phase=1, node=Node.END, min_bound=0, max_bound=np.inf)

        # Constraint arm positivity (prevent from local minimum with arms in the back)
        self.constraints.add(
            ConstraintFcn.TRACK_STATE, key="q", phase=self.takeoff, node=Node.END, index=3, min_bound=1.0, max_bound=np.inf
        )

        # Floor constraints for flat foot phases
        for p in self.flat_foot_phases:
            # Do not pull on floor
            for i in self.heel_and_toe_idx:
                self.constraints.add(
                    ConstraintFcn.TRACK_CONTACT_FORCES, phase=p, node=Node.ALL_SHOOTING, contact_index=i, max_bound=np.inf,
                )

            # Non-slipping constraints
            self.constraints.add(  # On only one of the feet
                ConstraintFcn.NON_SLIPPING,
                phase=p,
                node=Node.ALL_SHOOTING,
                normal_component_idx=(1, 2),
                tangential_component_idx=0,
                static_friction_coefficient=0.5
            )

        # Floor constraints for toe only phases
        for p in self.toe_only_phases:
            # Do not pull on floor
            for i in self.toe_idx:
                self.constraints.add(
                    ConstraintFcn.TRACK_CONTACT_FORCES, phase=p, node=Node.ALL_SHOOTING, contact_index=i, max_bound=np.inf,
                )

            # Non-slipping constraints
            self.constraints.add(  # On only one of the feet
                ConstraintFcn.NON_SLIPPING,
                phase=p,
                node=Node.ALL_SHOOTING,
                normal_component_idx=1,
                tangential_component_idx=0,
                static_friction_coefficient=0.5,
            )

    def _set_objective_functions(self):
        # Maximize the jump height
        self.objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_PREDICTED_COM_HEIGHT, weight=-100, phase=self.takeoff)

        # Minimize unnecessary movement during for the aerial and reception phases
        for p in range(2, 5):
            self.objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_STATE,
                key="qdot",
                derivative=True,
                weight=0.1,
                phase=p,
            )

        for i in range(self.n_phases):
            # Minimize time of the phase
            if self.time_min[i] != self.time_max[i]:
                self.objective_functions.add(
                    ObjectiveFcn.Mayer.MINIMIZE_TIME,
                    weight=0.1,
                    phase=i,
                    min_bound=self.time_min[i],
                    max_bound=self.time_max[i],
                )

    def _set_boundary_conditions(self):
        for i in range(self.n_phases):
            # Path constraints
            self.x_bounds.add(bounds=QAndQDotBounds(self.models[i], self.mapping_list))
            self.u_bounds.add([-500] * self.n_tau, [500] * self.n_tau)

        # Enforce the initial pose and velocity
        self.x_bounds[0][:, 0] = self.initial_states[:, 0]

        # Target the final pose (except for translation) and velocity
        self.objective_functions.add(
            ObjectiveFcn.Mayer.TRACK_STATE,
            key="q",
            phase=self.n_phases - 1,
            index=range(2, self.n_q),
            target=self.initial_states[2:self.n_q, :],
        )
        self.objective_functions.add(
            ObjectiveFcn.Mayer.TRACK_STATE,
            key="qdot",
            phase=self.n_phases - 1,
            target=self.initial_states[self.n_q:, :],
        )

    def _set_initial_guesses(self):
        for i in range(self.n_phases):
            self.x_init.add(self.initial_states)
            self.u_init.add([0] * self.n_tau)

    def _set_phase_transitions(self):
        # Phase transition
        self.phase_transitions.add(PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=0)
        self.phase_transitions.add(PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=1)
        self.phase_transitions.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=2)
        self.phase_transitions.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=3)

        # The end of the aerial
        self.constraints.add(toe_on_floor, phase=2, node=Node.END, min_bound=-0.001, max_bound=0.001, floor=self.floor)
        self.constraints.add(heel_on_floor, phase=3, node=Node.END, min_bound=-0.001, max_bound=0.001, floor=self.floor)

        # Allow for passive velocity at reception
        self.x_bounds[3].min[self.n_q :, 0] = 2 * self.x_bounds[3].min[self.n_q :, 0]
        self.x_bounds[3].max[self.n_q :, 0] = 2 * self.x_bounds[3].max[self.n_q :, 0]
        self.x_bounds[4].min[self.n_q :, 0] = 2 * self.x_bounds[4].min[self.n_q :, 0]
        self.x_bounds[4].max[self.n_q :, 0] = 2 * self.x_bounds[4].max[self.n_q :, 0]

    def solve(self, limit_memory_max_iter, exact_max_iter, load_path=None, force_no_graph=False, linear_solver="mumps"):
        def warm_start_nmpc(ocp, sol):
            state, ctrl, param = sol.states, sol.controls, sol.parameters
            u_init_guess = InitialGuessList()
            x_init_guess = InitialGuessList()
            for i in range(ocp.n_phases):
                u_init_guess.add(ctrl[i]["all"][:, :-1], interpolation=InterpolationType.EACH_FRAME)
                x_init_guess.add(state[i]["all"], interpolation=InterpolationType.EACH_FRAME)

            time_init_guess = InitialGuess(param["time"], name="time")
            ocp.update_initial_guess(x_init=x_init_guess, u_init=u_init_guess, param_init=time_init_guess)
            ocp.solver.set_lagrange_multiplier(sol)

        # Run optimizations
        if load_path:
            _, sol = OptimalControlProgram.load(load_path)
            return sol
        else:
            sol = None
            if limit_memory_max_iter > 0:
                sol = self.ocp.solve(
                    show_online_optim=exact_max_iter == 0 and not force_no_graph,
                    solver_options={
                        "linear_solver": linear_solver,
                        "hessian_approximation": "limited-memory",
                        "max_iter": limit_memory_max_iter,
                        "print_level": 0
                    },
                )
            if limit_memory_max_iter > 0 and exact_max_iter > 0:
                warm_start_nmpc(self.ocp, sol)
            if exact_max_iter > 0:
                sol = self.ocp.solve(
                    show_online_optim=True and not force_no_graph,
                    solver_options={
                        "linear_solver": linear_solver,
                        "hessian_approximation": "exact",
                        "max_iter": exact_max_iter,
                        "warm_start_init_point": "yes",
                        "print_level": 0
                    },
                )

            return sol
