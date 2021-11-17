import biorbd_casadi as biorbd
from scipy import optimize
import numpy as np
from casadi import MX
from bioptim import BiMapping, OdeSolver


class Jumper:
    model_files = (
        "jumper2contacts.bioMod",
        "jumper1contacts.bioMod",
        "jumper1contacts.bioMod",
        "jumper1contacts.bioMod",
        "jumper2contacts.bioMod",
    )
    time_min = 0.2, 0.05, 0.6, 0.05, 0.1
    time_max = 0.5, 0.5, 2.0, 0.5, 0.5
    phase_time = 0.3, 0.2, 0.6, 0.2, 0.2
    n_shoot = 30, 15, 20, 30, 30
    ode_solver = None
    n_max_iter_limited_memory = None
    n_max_iter_exact = None

    q_mapping = BiMapping(
        [0, 1, 2, 3, 3, 4, 5, 6, 4, 5, 6],
        [0, 1, 2, 3, 5, 6, 7]
    )
    tau_mapping = BiMapping(
        [None, None, None, 0, 0, 1, 2, 3, 1, 2, 3],
        [3, 5, 6, 7]
    )
    tau_constant_bound = 500
    initial_states = []
    body_at_first_node = [0, 0, 0, 2.10, 1.15, 0.80, 0.20]
    initial_velocity = [0, 0, 0, 0, 0, 0, 0]
    tau_min = 20  # Tau minimal bound despite the torque activation
    arm_dof = 3
    heel_dof = 6
    heel_marker_idx = 85
    toe_marker_idx = 86

    floor_z = 0.0
    flat_foot_phases = 0, 4  # The indices of flat foot phases
    toe_only_phases = 1, 3  # The indices of toe only phases

    flatfoot_contact_x_idx = ()  # Contacts indices of heel and toe in bioMod 2 contacts
    flatfoot_contact_y_idx = (1, 4)  # Contacts indices of heel and toe in bioMod 2 contacts
    flatfoot_contact_z_idx = (0, 2, 3, 5)  # Contacts indices of heel and toe in bioMod 2 contacts
    flatfoot_non_slipping = ((1,), (0, 2))  # (X-Y components), Z components

    toe_contact_x_idx = ()  # Contacts indices of toe in bioMod 1 contact
    toe_contact_y_idx = (0, 2)  # Contacts indices of toe in bioMod 1 contact
    toe_contact_z_idx = (1, 2)  # Contacts indices of toe in bioMod 1 contact
    toe_non_slipping = ((0,), 1)  # (X-Y components), Z components
    static_friction_coefficient = 0.5

    def __init__(self, path_to_models):
        self.models = [biorbd.Model(path_to_models + "/" + elt) for elt in self.model_files]

    def find_initial_root_pose(self):
        # This method finds a root pose such that the body of a given pose has its CoM centered to the feet
        model = self.models[0]
        n_root = model.nbRoot()

        body_pose_no_root = self.q_mapping.to_second.map(self.body_at_first_node)[n_root:, 0]
        bimap = BiMapping(list(range(n_root)) + [None] * body_pose_no_root.shape[0], list(range(n_root)))

        bound_min = []
        bound_max = []
        for i in range(model.nbSegment()):
            seg = model.segment(i)
            for r in seg.QRanges():
                bound_min.append(r.min())
                bound_max.append(r.max())
        bound_min = bimap.to_first.map(np.array(bound_min)[:, np.newaxis])
        bound_max = bimap.to_first.map(np.array(bound_max)[:, np.newaxis])
        root_bounds = (list(bound_min[:, 0]), list(bound_max[:, 0]))

        q_sym = MX.sym("Q", model.nbQ(), 1)
        com_func = biorbd.to_casadi_func("com", model.CoM, q_sym)
        contacts_func = biorbd.to_casadi_func("contacts", model.constraintsInGlobal, q_sym, True)
        shoulder_jcs_func = biorbd.to_casadi_func("shoulder_jcs", model.globalJCS, q_sym, 3)
        hand_marker_func = biorbd.to_casadi_func("hand_marker", model.marker, q_sym, 32)

        def objective_function(q_root, *args, **kwargs):
            # Center of mass
            q = bimap.to_second.map(q_root[:, np.newaxis])[:, 0]
            q[model.nbRoot():] = body_pose_no_root
            com = np.array(com_func(q))
            contacts = np.array(contacts_func(q))
            mean_contacts = np.mean(contacts, axis=1)
            shoulder_jcs = np.array(shoulder_jcs_func(q))
            hand = np.array(hand_marker_func(q))

            # Prepare output
            out = np.ndarray((0,))

            # The center of contact points should be at 0
            out = np.concatenate((out, mean_contacts[0:2]))
            out = np.concatenate((out, contacts[2, self.flatfoot_contact_z_idx]))

            # The projection of the center of mass should be at 0 and at 0.95 meter high
            out = np.concatenate((out, (com + np.array([[0, 0, -0.95]]).T)[:, 0]))

            # Keep the arms horizontal
            out = np.concatenate((out, (shoulder_jcs[2, 3] - hand[2])))

            return out

        q_root0 = np.mean(root_bounds, axis=0)
        pos = optimize.least_squares(objective_function, x0=q_root0, bounds=root_bounds)
        root = np.zeros(n_root)
        root[bimap.to_first.map_idx] = pos.x
        return root

    def show(self, q):
        import bioviz
        b = bioviz.Viz(self.models[0].path().absolutePath().to_string())
        b.set_q(q if len(q.shape) == 1 else q[:, 0])
        b.exec()


class JumperRK4(Jumper):
    n_shoot = 25, 10, 20, 30, 30
    n_max_iter_limited_memory = 75
    n_max_iter_exact = 1000

    def __init__(self, path_to_models):
        super(JumperRK4, self).__init__(path_to_models)
        self.ode_solver = OdeSolver.RK4()


class JumperCOLLOCATION(Jumper):
    n_shoot = 25, 10, 20, 30, 30
    n_max_iter_limited_memory = 175
    n_max_iter_exact = 1000

    def __init__(self, path_to_models):
        super(JumperCOLLOCATION, self).__init__(path_to_models)
        self.ode_solver = OdeSolver.COLLOCATION()
