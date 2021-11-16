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
    tau_min = 15
    initial_states = []
    body_at_first_node = [0, 0, 0, 0.7, -0.9, 0.50]
    initial_velocity = [0, 0, 0, 0, 0, 0]
    heel_idx = 0
    toe_idx = 1
    floor_z = 0.10

    def __init__(self, path_to_models):
        self.path = path_to_models
        self.model = biorbd.Model(path_to_models + "/" + self.model_files)

    def find_initial_root_pose(self):
        model = biorbd.Model(self.path + "/" + self.model_files)
        # This method finds a root pose such that the body of a given pose has its CoM centered to the feet
        n_root = model.nbRoot()
        body_pose_no_root = self.body_at_first_node[n_root:]

        bound_min = []
        bound_max = []
        for i in range(model.nbSegment()):
            seg = model.segment(i)
            for r in seg.QRanges():
                bound_min.append(r.min())
                bound_max.append(r.max())
        bound_max[-1] = 0.6
        bounds = (bound_min, bound_max)

        q_sym = MX.sym("Q", model.nbQ(), 1)
        qdot_sym = MX.sym("Qdot", model.nbQ(), 1)
        qddot_sym = MX.sym("Qddot", model.nbQ(), 1)
        tau_sym = MX.sym("Tau", model.nbQ(), 1)
        com_func = biorbd.to_casadi_func("com", model.CoM, q_sym)
        fd_func = biorbd.to_casadi_func("fd", model.ForwardDynamics, q_sym, qdot_sym, tau_sym)
        marker_func = biorbd.to_casadi_func("markers", model.markers, q_sym, True)
        marker_accel_func = biorbd.to_casadi_func("marker_accel", model.markerAcceleration, q_sym, qdot_sym, qddot_sym, True)

        def objective_function(q, *args, **kwargs):
            # Center of mass
            com = np.array(com_func(q))[1:, 0]  # Y and Z
            contacts = np.array(marker_func(q)[1:, :])  # Y and Z
            mean_contacts = np.mean(contacts, axis=1)

            # Prepare output
            out = np.ndarray((0,))

            # The center of contact points and the COM should be at 0
            out = np.concatenate((out, mean_contacts[0][np.newaxis]))
            out = np.concatenate((out, contacts[1, :] - self.floor_z))

            # The projection of the center of mass should be at 0 and at 0.95 meter high
            out = np.concatenate((out, (com - [mean_contacts[0], 0.75]) * 10))

            tau = np.zeros(model.nbQ(),)
            qdot = np.zeros(model.nbQ(),)
            qddot = fd_func(q, qdot, tau)
            out = np.concatenate((out, np.array(marker_accel_func(q, qdot, qddot))[2, :]))

            return out

        q0 = np.mean(bounds, axis=0)
        q0[n_root:] = body_pose_no_root
        pos = optimize.least_squares(objective_function, x0=q0, bounds=bounds)
        return pos.x[:, np.newaxis]

    def show(self, q):
        import bioviz
        b = bioviz.Viz(self.model.path().absolutePath().to_string())
        b.set_q(q if len(q.shape) == 1 else q[:, 0])
        b.exec()
