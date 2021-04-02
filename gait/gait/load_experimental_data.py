from ezc3d import c3d
import numpy as np


class C3dData:
    def __init__(self, file_path):
        self.c3d = c3d(file_path, extract_forceplat_data=True)
        self.marker_names = [
            "L_IAS",
            "L_IPS",
            "R_IPS",
            "R_IAS",
            "R_FTC",
            "R_Thigh_Top",
            "R_Thigh_Down",
            "R_Thigh_Front",
            "R_Thigh_Back",
            "R_FLE",
            "R_FME",
            "R_FAX",
            "R_TTC",
            "R_Shank_Top",
            "R_Shank_Down",
            "R_Shank_Front",
            "R_Shank_Tibia",
            "R_FAL",
            "R_TAM",
            "R_FCC",
            "R_FM1",
            "R_FMP1",
            "R_FM2",
            "R_FMP2",
            "R_FM5",
            "R_FMP5",
        ]

        self.trajectories = self.get_marker_trajectories(self.c3d, self.marker_names)
        self.forces = self.get_forces(self.c3d)
        self.moments = self.get_moment(self.c3d)
        self.cop = self.get_cop(self.c3d)
        self.events = self.get_event_rhs_rto(self.c3d)
        self.indices = self.get_indices()
        self.phase_time = self.get_time()

    @staticmethod
    def get_marker_trajectories(loaded_c3d, marker_names):
        """
        get markers trajectories
        """

        # LOAD C3D FILE
        points = loaded_c3d["data"]["points"]
        labels_markers = loaded_c3d["parameters"]["POINT"]["LABELS"]["value"]

        # GET THE MARKERS POSITION (X, Y, Z) AT EACH POINT
        markers = np.zeros((3, len(marker_names), len(points[0, 0, :])))

        # pelvis markers
        for i, name in enumerate(marker_names):
            markers[:, i, :] = points[:3, labels_markers.index(name), :] * 1e-3
        return markers

    @staticmethod
    def get_forces(loaded_c3d):
        """
        get ground reaction forces from force platform
        """
        platform = loaded_c3d["data"]["platform"][0]
        return platform["force"]

    @staticmethod
    def get_moment(loaded_c3d):
        """
        get moments value expressed at the center of pression
        from force platform
        """
        platform = loaded_c3d["data"]["platform"][0]
        return platform["Tz"] * 1e-3

    @staticmethod
    def get_cop(loaded_c3d):
        """
        get the trajectory of the center of pressure (cop)
        from force platform
        """
        platform = loaded_c3d["data"]["platform"][0]
        return platform["center_of_pressure"] * 1e-3

    @staticmethod
    def get_event_rhs_rto(loaded_c3d):
        """
        find event from c3d file : heel strike (HS) and toe off (TO)
        determine the indexes of the beginning and end of the cycle
        """

        time = loaded_c3d["parameters"]["EVENT"]["TIMES"]["value"][1, :]
        labels_time = loaded_c3d["parameters"]["EVENT"]["LABELS"]["value"]

        def get_indices(name, time):
            return [i for (y, i) in zip(time, range(len(time))) if name == y]

        rhs = time[get_indices("RHS", labels_time)]
        rto = time[get_indices("RTO", labels_time)]
        if len(rto) > 1:
            rto = max(rto)
        else:
            rto = rto[0]

        return rhs, rto

    def get_indices(self):
        """
        find phase indexes
        indexes corresponding to the event that defines phases :
        - start : heel strike
        - 2 contacts : toes on the ground
        - heel rise : rising of the heel
        - stop stance : foot off the ground
        - stop : second heel strike
        """
        freq = self.c3d["parameters"]["POINT"]["RATE"]["value"][0]
        threshold = 0.04

        # get events for start and stop of the cycle
        rhs, rto = C3dData.get_event_rhs_rto(self.c3d)
        idx_start = int(round(rhs[0] * freq) + 1)
        idx_stop_stance = int(round(rto * freq) + 1)
        idx_stop = int(round(rhs[1] * freq) + 1)

        # get markers position
        markers = C3dData.get_marker_trajectories(self.c3d, self.marker_names)
        heel = markers[:, 19, idx_start:idx_stop_stance]
        meta1 = markers[:, 20, idx_start:idx_stop_stance]
        meta5 = markers[:, 24, idx_start:idx_stop_stance]

        # Heel rise
        idx_heel = np.where(heel[2, :] > threshold)
        idx_heel_rise = idx_start + int(idx_heel[0][0])

        # forefoot
        idx_meta1 = np.where(meta1[2, :] < threshold)
        idx_meta5 = np.where(meta5[2, :] < threshold)
        idx_2_contacts = idx_start + np.max([idx_meta5[0][0], idx_meta1[0][0]])
        return [idx_start, idx_2_contacts, idx_heel_rise, idx_stop_stance, idx_stop]

    def get_time(self):
        """
        find phase duration
        """
        freq = self.c3d["parameters"]["ANALOG"]["RATE"]["value"][0]

        index = self.get_indices()
        phase_time = []
        for i in range(len(index) - 1):
            phase_time.append((1 / freq * (index[i + 1] - index[i] + 1)))
        return phase_time


class LoadData:
    def __init__(self, model, c3d_file, q_file, qdot_file):
        def load_txt_file(file_path, size):
            data_tp = np.loadtxt(file_path)
            nb_frame = int(len(data_tp) / size)
            out = np.zeros((size, nb_frame))
            for n in range(nb_frame):
                out[:, n] = data_tp[n * size : n * size + size]
            return out

        self.model = model
        self.nb_q = model.nbQ()
        self.nb_qdot = model.nbQdot()
        self.nb_markers = model.nbMarkers()

        # files path
        self.c3d_data = C3dData(c3d_file)
        self.q = load_txt_file(q_file, self.nb_q)
        self.qdot = load_txt_file(qdot_file, self.nb_qdot)

    def dispatch_data(self, data, nb_shooting):
        """
        divide and adjust data dimensions to match number of shooting point for each phase
        """

        index = self.c3d_data.get_indices()
        out = []
        for i in range(len(nb_shooting)):
            a = (index[i + 1] + 1 - index[i]) / (nb_shooting[i] + 1)
            if len(data.shape) == 3:
                if a.is_integer():
                    x = data[:, :, index[i] : index[i + 1] + 1]
                    out.append(x[:, :, 0 :: int(a)])
                else:
                    x = data[:, :, index[i] : index[i + 1]]
                    out.append(x[:, :, 0 :: int(a)])

            else:
                if a.is_integer():
                    x = data[:, index[i] : index[i + 1] + 1]
                    out.append(x[:, 0 :: int(a)])
                else:
                    x = data[:, index[i] : index[i + 1]]
                    out.append(x[:, 0 :: int(a)])
        return out
