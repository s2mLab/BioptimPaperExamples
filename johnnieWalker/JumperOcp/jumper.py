import biorbd_casadi as biorbd
from scipy import optimize
import numpy as np
from casadi import MX
from bioptim import BiMapping


class Jumper:
    model_files = "jumperSoftContacts.bioMod"
    time_min = 1.5
    time_max = 1.5
    phase_time = 1.5
    n_shoot = 300

    tau_constant_bound = 500
    initial_states = []
    body_at_first_node = [0, 0, 0, 0.7, -0.9, 0.50]
    initial_velocity = [0, 0, 0, 0, 0, 0]
    heel_idx = 0
    toe_idx = 1
    floor_z = 0.18

    def __init__(self, path_to_models):
        self.model = biorbd.Model(path_to_models + "/" + self.model_files)

    def find_initial_root_pose(self):
        # This method finds a root pose such that the body of a given pose has its CoM centered to the feet
        n_root = self.model.nbRoot()
        body_pose_no_root = self.body_at_first_node[n_root:]

        bound_min = []
        bound_max = []
        for i in range(self.model.nbSegment()):
            seg = self.model.segment(i)
            for r in seg.QRanges():
                bound_min.append(r.min())
                bound_max.append(r.max())
        root_bounds = (bound_min[:n_root], bound_max[:n_root])

        q_sym = MX.sym("Q", self.model.nbQ(), 1)
        com_func = biorbd.to_casadi_func("com", self.model.CoM, q_sym)
        marker_func = biorbd.to_casadi_func("markers", self.model.markers, q_sym, True)

        def objective_function(q_root, *args, **kwargs):
            # Center of mass
            q = np.concatenate((q_root, body_pose_no_root))
            com = np.array(com_func(q))[1:, 0]  # Y and Z
            contacts = np.array(marker_func(q)[1:, :])  # Y and Z
            mean_contacts = np.mean(contacts, axis=1)

            # Prepare output
            out = np.ndarray((0,))

            # The center of contact points should be at 0
            out = np.concatenate((out, mean_contacts[0][np.newaxis]))
            out = np.concatenate((out, contacts[1, :] - self.floor_z))

            # The projection of the center of mass should be at 0 and at 0.95 meter high
            out = np.concatenate((out, com - [0, 0.95]))

            return out

        q_root0 = np.mean(root_bounds, axis=0)
        pos = optimize.least_squares(objective_function, x0=q_root0, bounds=root_bounds)
        return pos.x

    def show(self, q):
        import bioviz
        b = bioviz.Viz(self.model.path().absolutePath().to_string())
        b.set_q(q if len(q.shape) == 1 else q[:, 0])
        b.exec()
