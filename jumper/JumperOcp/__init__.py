import numpy as np
from bioptim import (
    OdeSolver,
    Node,
    OptimalControlProgram,
    ConstraintFcn,
    ObjectiveFcn,
    DynamicsFcn,
    QAndQDotBounds,
    PhaseTransitionFcn,
    ConstraintList,
    ObjectiveList,
    DynamicsList,
    BoundsList,
    InitialGuessList,
    PhaseTransitionList,
    BiMappingList,
    ControlType,
)

from .jumper import Jumper
from .penalty_functions import com_dot_z, marker_on_floor, contact_force_continuity
from .viz import add_custom_plots


class JumperOcp:
    n_q, n_qdot, n_tau = -1, -1, -1
    mapping_list = BiMappingList()

    dynamics = DynamicsList()
    constraints = ConstraintList()
    objective_functions = ObjectiveList()
    x_bounds = BoundsList()
    u_bounds = BoundsList()
    phase_transitions = PhaseTransitionList()
    initial_states = []
    x_init = InitialGuessList()
    u_init = InitialGuessList()

    def __init__(self, jumper: Jumper, n_phases, n_thread=8, control_type=ControlType.CONSTANT):
        if n_phases < 1 or n_phases > 5:
            raise ValueError("n_phases must be comprised between 1 and 5")
        self.jumper = jumper
        self.n_phases = n_phases
        self.takeoff = 0 if self.n_phases == 1 else 1  # The index of takeoff phase
        self.control_type = control_type
        self.control_nodes = Node.ALL if self.control_type == ControlType.LINEAR_CONTINUOUS else Node.ALL_SHOOTING

        self._set_dimensions_and_mapping()
        self._set_initial_states()

        self._set_dynamics()
        self._set_constraints()
        self._set_objective_functions()

        self._set_boundary_conditions()
        self._set_phase_transitions()

        self._set_initial_guesses()

        self.ocp = OptimalControlProgram(
            self.jumper.models[:self.n_phases],
            self.dynamics,
            self.jumper.n_shoot[:self.n_phases],
            self.jumper.phase_time[:self.n_phases],
            x_init=self.x_init,
            x_bounds=self.x_bounds,
            u_init=self.u_init,
            u_bounds=self.u_bounds,
            objective_functions=self.objective_functions,
            constraints=self.constraints,
            variable_mappings=self.mapping_list,
            phase_transitions=self.phase_transitions,
            n_threads=n_thread,
            control_type=self.control_type,
            ode_solver=OdeSolver.RK8()
        )

    def _set_initial_states(self):
        initial_pose = np.array([self.jumper.body_at_first_node]).T
        initial_velocity = np.array([self.jumper.initial_velocity]).T

        initial_pose[:self.jumper.models[0].nbRoot(), 0] = self.jumper.find_initial_root_pose()

        self.initial_states = np.concatenate((initial_pose, initial_velocity))

    def _set_dimensions_and_mapping(self):
        self.mapping_list.add("q", self.jumper.q_mapping.to_second.map_idx, self.jumper.q_mapping.to_first.map_idx)
        self.mapping_list.add("qdot", self.jumper.q_mapping.to_second.map_idx, self.jumper.q_mapping.to_first.map_idx)
        self.mapping_list.add("tau", self.jumper.tau_mapping.to_second.map_idx, self.jumper.tau_mapping.to_first.map_idx)
        self.n_q = len(self.mapping_list["q"].to_first)
        self.n_qdot = self.n_q
        self.n_tau = len(self.mapping_list["tau"].to_first)

    def _set_dynamics(self):
        self.dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True)  # Flat foot
        self.dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True)  # Toe only
        self.dynamics.add(DynamicsFcn.TORQUE_DRIVEN)  # Aerial phase
        self.dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True)  # Toe only
        self.dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True)  # Flat foot

    def _set_constraints(self):
        # Torque constrained to torqueMax
        for i in range(self.n_phases):
            self.constraints.add(ConstraintFcn.TORQUE_MAX_FROM_Q_AND_QDOT, phase=i, node=self.control_nodes, min_torque=self.jumper.tau_min)

        # Positivity of CoM_dot on z axis prior the take-off
        self.constraints.add(com_dot_z, phase=self.takeoff, node=Node.END, min_bound=0, max_bound=np.inf)

        # Constraint arm positivity (prevent from local minimum with arms in the back)
        self.constraints.add(
            ConstraintFcn.TRACK_STATE, key="q", phase=self.takeoff, node=Node.END, index=3, min_bound=0, max_bound=np.inf
        )

        # Floor constraints for flat foot phases
        for p in self.jumper.flat_foot_phases:
            if p >= self.n_phases:
                break

            # Do not pull on floor
            for i in self.jumper.flatfoot_contact_z_idx:
                self.constraints.add(
                    ConstraintFcn.TRACK_CONTACT_FORCES, phase=p, node=self.control_nodes, contact_index=i, max_bound=np.inf
                )

            # Non-slipping constraints
            self.constraints.add(  # On only one of the feet
                ConstraintFcn.NON_SLIPPING,
                phase=p,
                node=self.control_nodes,
                tangential_component_idx=self.jumper.flatfoot_non_slipping[0],
                normal_component_idx=self.jumper.flatfoot_non_slipping[1],
                static_friction_coefficient=self.jumper.static_friction_coefficient,
            )

        # Floor constraints for toe only phases
        for p in self.jumper.toe_only_phases:
            if p >= self.n_phases:
                break

            # Do not pull on floor
            for i in self.jumper.toe_contact_z_idx:
                self.constraints.add(
                    ConstraintFcn.TRACK_CONTACT_FORCES, phase=p, node=self.control_nodes, contact_index=i, max_bound=np.inf
                )

            # The heel must remain over floor
            self.constraints.add(
                marker_on_floor,
                phase=p,
                node=Node.ALL,
                index=2,
                min_bound=-0.0001,
                max_bound=np.inf,
                marker=self.jumper.heel_marker_idx,
                target=self.jumper.floor_z,
            )

            # Non-slipping constraints
            self.constraints.add(  # On only one of the feet
                ConstraintFcn.NON_SLIPPING,
                phase=p,
                node=self.control_nodes,
                tangential_component_idx=self.jumper.toe_non_slipping[0],
                normal_component_idx=self.jumper.toe_non_slipping[1],
                static_friction_coefficient=self.jumper.static_friction_coefficient,
            )

    def _set_objective_functions(self):
        # Maximize the jump height
        self.objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_PREDICTED_COM_HEIGHT, weight=-100, phase=self.takeoff)

        # Minimize the tau on root if present
        for p in range(self.n_phases):
            root = [i for i in self.jumper.tau_mapping.to_second.map_idx[:self.jumper.models[p].nbRoot()] if i is not None]
            if root:
                self.objective_functions.add(
                    ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
                    key="tau",
                    weight=0.1,
                    phase=p,
                    index=root,
                )

        # Minimize unnecessary acceleration during for the aerial and reception phases
        for p in range(self.n_phases):
            if self.control_type == ControlType.LINEAR_CONTINUOUS:
                self.objective_functions.add(
                    ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
                    key="tau",
                    weight=0.1,
                    derivative=True,
                    phase=p,
                )

        for p in range(2, self.n_phases):
            self.objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_STATE,
                key="qdot",
                weight=0.1,
                derivative=True,
                phase=p,
            )

        # Minimize time of the phase
        for i in range(self.n_phases):
            if self.jumper.time_min[i] != self.jumper.time_max[i]:
                self.objective_functions.add(
                    ObjectiveFcn.Mayer.MINIMIZE_TIME,
                    weight=0.1,
                    phase=i,
                    min_bound=self.jumper.time_min[i],
                    max_bound=self.jumper.time_max[i],
                )

    def _set_boundary_conditions(self):
        for i in range(self.n_phases):
            # Path constraints
            self.x_bounds.add(
                bounds=QAndQDotBounds(self.jumper.models[i], dof_mappings=self.mapping_list)
            )
            if i == 3 or i == 4:
                # Allow greater speed in passive reception
                self.x_bounds[i].max[self.jumper.heel_dof + self.n_q, :] *= 2
            self.u_bounds.add([-self.jumper.tau_constant_bound] * self.n_tau, [self.jumper.tau_constant_bound] * self.n_tau)

        # Enforce the initial pose and velocity
        self.x_bounds[0][:, 0] = self.initial_states[:, 0]

        # Target the final pose (except for translation)
        if self.n_phases >= 4:
            trans_root = self.jumper.models[self.n_phases-1].segment(0).nbDofTrans()
            self.constraints.add(
                ConstraintFcn.TRACK_STATE,
                key="q",
                node=Node.END,
                phase=self.n_phases - 1,
                index=range(trans_root, self.n_q),
                target=self.initial_states[trans_root:self.n_q, :],
                min_bound=-0.1,
                max_bound=0.1,
            )
            self.constraints.add(
                ConstraintFcn.TRACK_STATE,
                key="qdot",
                node=Node.END,
                phase=self.n_phases - 1,
                target=self.initial_states[self.n_q:, :],
                min_bound=-0.1,
                max_bound=0.1,
            )

    def _set_initial_guesses(self):
        for i in range(self.n_phases):
            self.x_init.add(self.initial_states)
            self.u_init.add([0] * self.n_tau)

    def _set_phase_transitions(self):
        if self.n_phases >= 2:  # 2 contacts to 1 contact
            self.phase_transitions.add(PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=0)
        if self.n_phases >= 3:  # 1 contact to aerial
            self.phase_transitions.add(PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=1)
        if self.n_phases >= 4:  # aerial to 1 contact
            self.phase_transitions.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=2)
        if self.n_phases >= 5:  # 1 contact to 2 contacts
            self.phase_transitions.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=3)

        # if self.n_phases >= 2:  # The contact forces at the end of flat foot equal the beginning of the next phase
        #     links_0_to_1 = ((0, None), (1, None), (2, 0), (3, 1), (3, 2))
        #     links_1_to_2 = ((0, None), (1, None), (2, None))
        #     for link in links_0_to_1:
        #         self.constraints.add(
        #             contact_force_continuity,
        #             phase=0,
        #             node=Node.TRANSITION,
        #             idx_pre=link[0],
        #             idx_post=link[1],
        #         )
        #
        #     for link in links_1_to_2:
        #         self.constraints.add(
        #             contact_force_continuity,
        #             phase=1,
        #             node=Node.TRANSITION,
        #             idx_pre=link[0],
        #             idx_post=link[1],
        #         )

        if self.n_phases >= 3:  # The end of the aerial
            self.constraints.add(
                marker_on_floor,
                phase=2,
                index=2,
                node=Node.END,
                min_bound=-0.001,
                max_bound=0.001,
                marker=self.jumper.toe_marker_idx,
                target=self.jumper.floor_z,
            )
        if self.n_phases >= 4:  # 2 contacts on floor
            self.constraints.add(
                marker_on_floor,
                phase=3,
                index=2,
                node=Node.END,
                min_bound=-0.001,
                max_bound=0.001,
                marker=self.jumper.heel_marker_idx,
                target=self.jumper.floor_z,
            )

        # Allow for passive velocity at reception
        if self.n_phases >= 4:
            self.x_bounds[3].min[self.n_q :, 0] = 2 * self.x_bounds[3].min[self.n_q :, 0]
            self.x_bounds[3].max[self.n_q :, 0] = 2 * self.x_bounds[3].max[self.n_q :, 0]
        if self.n_phases >= 5:
            self.x_bounds[4].min[self.n_q :, 0] = 2 * self.x_bounds[4].min[self.n_q :, 0]
            self.x_bounds[4].max[self.n_q :, 0] = 2 * self.x_bounds[4].max[self.n_q :, 0]

    def solve(self, limit_memory_max_iter, exact_max_iter, load_path=None, force_no_graph=False, linear_solver="mumps"):
        # Run optimizations
        if not force_no_graph:
            add_custom_plots(self.ocp, self)

        if load_path:
            _, sol = OptimalControlProgram.load(load_path)
            return sol
        else:
            sol = None
            if limit_memory_max_iter > 0:
                sol = self.ocp.solve(
                    show_online_optim=exact_max_iter == 0 and not force_no_graph,
                    solver_options={
                        "hessian_approximation": "limited-memory",
                        "max_iter": limit_memory_max_iter,
                        "linear_solver": linear_solver
                    },
                )
            if limit_memory_max_iter > 0 and exact_max_iter > 0:
                self.ocp.set_warm_start(sol)
            if exact_max_iter > 0:
                sol = self.ocp.solve(
                    show_online_optim=True and not force_no_graph,
                    solver_options={
                        "hessian_approximation": "exact",
                        "max_iter": exact_max_iter,
                        "warm_start_init_point": "yes",
                        "linear_solver": "ma57",
                    },
                )

            return sol
