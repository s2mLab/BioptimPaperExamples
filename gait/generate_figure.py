from time import time

import matplotlib.pyplot as plt
import numpy as np
import bioviz
import biorbd_casadi as biorbd
from bioptim import OptimalControlProgram, Shooting

from gait.load_experimental_data import LoadData
from gait.ocp import get_phase_time_shooting_numbers, get_experimental_data

def get_contact_index(nlp, tag):
    force_names = [s.to_string() for s in nlp.model.contactNames()]
    return [i for i, t in enumerate([s[-1] == tag for s in force_names]) if t]

def get_markers_index(nlp, tag):
    force_names = [s.to_string() for s in nlp.model.contactNames()]
    marker_names = [s.to_string() for s in nlp.model.markerNames()]

    contact_name = [force_names[i] for i, t in enumerate([s[-1] == tag for s in force_names]) if t]
    return [np.vstack([i for i, t in enumerate([s == c[:-2] for s in marker_names]) if t]) for c in contact_name]

def compute_contact_forces(states, controls, nlp):
    force = np.ndarray((nlp.model.nbContacts(), states.shape[1] - 1))
    for n in range(states.shape[1] - 1):
        force[:, n] = np.array(nlp.contact_forces_func(states[:, n], controls[:, n], 0)).squeeze()
    return force

def compute_sum_forces(states, controls, nlp):
    force = compute_contact_forces(states, controls, nlp)
    grf = np.vstack([np.sum(force[get_contact_index(nlp, "X"), :], axis=0),
                     np.sum(force[get_contact_index(nlp, "Y"), :], axis=0),
                     np.sum(force[get_contact_index(nlp, "Z"), :], axis=0)])
    return grf

def compute_cop(states, controls, nlp, markers_idx):
    force = compute_contact_forces(states, controls, nlp)
    force_z = force[get_contact_index(nlp, "Z"), :]
    markers = compute_markers_position(states[:nlp.model.nbQ(), :], nlp)[:, :, :-1]
    cop = np.vstack([np.sum(np.vstack([markers[0, m] * force_z[i] for (i,m) in enumerate(markers_idx)]), axis=0) / np.sum(force_z, axis=0), # My/Fz (-x*fz/Fz)
                     np.sum(np.vstack([markers[1, m] * force_z[i] for (i,m) in enumerate(markers_idx)]), axis=0) / np.sum(force_z, axis=0), # Mx/Fz (y*fz/Fz)
                     np.zeros(states.shape[1] - 1)])
    return cop

def compute_moments(states, controls, nlp, markers_idx):
    force = compute_contact_forces(states, controls, nlp)
    force_x = force[get_contact_index(nlp, "X"), :]
    force_y = force[get_contact_index(nlp, "Y"), :]
    force_z = force[get_contact_index(nlp, "Z"), :]
    markers = compute_markers_position(states[:nlp.model.nbQ(), :], nlp)[:, markers_idx, :-1].squeeze()
    cop = compute_cop(states, controls, nlp, markers_idx)
    markers_cop = markers - cop

    idx_x = get_markers_index(nlp, "X")
    idx_y = get_markers_index(nlp, "Y")
    moments = np.vstack([np.sum(np.vstack([markers_cop[1, i] * force_z[i] for i in range(len(markers_idx))]), axis=0),  # y*fz
                         np.sum(np.vstack([-markers_cop[0, i] * force_z[i] for i in range(len(markers_idx))]), axis=0),  # -x*fz
                         np.sum(np.vstack([markers_cop[0, markers_idx.index(y)] * force_y[i] for (i, y) in enumerate(idx_y)]), axis=0) - np.sum(np.vstack([markers_cop[1, markers_idx.index(x)] * force_x[i] for (i, x) in enumerate(idx_x)]), axis=0)])  # x*fy - y*fx
    return moments

def compute_markers_position(q, nlp):
    markers_positions = np.ndarray((3, nlp.model.nbMarkers(), q.shape[1]))
    markers = biorbd.to_casadi_func("markers", ocp_previous.nlp[p].model.markers, ocp_previous.nlp[p].states["q"].mx)
    for n in range(q.shape[1]):
        markers_positions[:, :, n] = np.array(markers(q[:, n])).squeeze()
    return markers_positions

def rmse(data, data_ref, grf=False):
    rmse = np.sqrt(((data - data_ref[:, :-1]) ** 2).mean()) if grf else np.sqrt(((data - data_ref) ** 2).mean())
    return rmse

def plot_contact_forces(grf_ref, grf):
    n_shoot=0
    plt.figure()
    for p in range(len(grf)):
        t=np.linspace(n_shoot, n_shoot + grf_ref[p].shape[1], grf_ref[p].shape[1])
        plt.plot(t, grf_ref[p].T, 'k')
        plt.plot(t[:-1], grf[p][0, :], 'b')
        plt.plot(t[:-1], grf[p][1, :], 'g')
        plt.plot(t[:-1], grf[p][2, :], 'r')
        n_shoot+=grf_ref[p].shape[1]
    plt.show()


# Define the problem -- model path
root_path = "/".join(__file__.split("/")[:-1]) + "/"
biorbd_model = (
    biorbd.Model(root_path + "models/Gait_1leg_12dof_heel.bioMod"),
    biorbd.Model(root_path + "models/Gait_1leg_12dof_flatfoot.bioMod"),
    biorbd.Model(root_path + "models/Gait_1leg_12dof_forefoot.bioMod"),
    biorbd.Model(root_path + "models/Gait_1leg_12dof_0contact.bioMod"),
)

# Generate data from file
# --- files path ---
c3d_file = root_path + "data/normal01_out.c3d"
q_kalman_filter_file = root_path + "data/normal01_q_KalmanFilter.txt"
qdot_kalman_filter_file = root_path + "data/normal01_qdot_KalmanFilter.txt"
data = LoadData(biorbd_model[0], c3d_file, q_kalman_filter_file, qdot_kalman_filter_file)

# --- phase time and number of shooting ---
phase_time, number_shooting_points = get_phase_time_shooting_numbers(data, 0.01)
# --- get experimental data ---
q_ref, qdot_ref, markers_ref, grf_ref, moments_ref, cop_ref = get_experimental_data(data, number_shooting_points, phase_time)
m = markers_ref[0][:, :, :-1]
for i in range(1, len(markers_ref)):
    m = np.concatenate([m, markers_ref[i][:, :, :-1]], axis=2)

# Load previous solution
ocp_previous, sol_previous=OptimalControlProgram.load(root_path + "gait_cv.bo")

# --- plot graphs & animate ---
# sol_merged = sol_previous.merge_phases()
# q_merged = sol_merged.states['q']
# b = bioviz.Viz(model_path=root_path + "models/Gait_1leg_12dof_flatfoot.bioMod")
# b.load_movement(q_merged)
# b.load_experimental_markers(m)
# b.exec()
# sol_previous.animate()
# sol_previous.graphs()

# --- compute single shooting error ---
sol_ss = sol_previous.integrate(shooting_type=Shooting.SINGLE_CONTINUOUS, merge_phases=False)
ss_err_trans = np.sqrt(np.mean((sol_ss.states[-1]["q"][:3, -1] - sol_previous.states[-1]["q"][:3, -1]) ** 2))
ss_err_rot = np.sqrt(np.mean((sol_ss.states[-1]["q"][3:, -1] - sol_previous.states[-1]["q"][3:, -1]) ** 2))

print("*********************************************")
print(f"Single shooting error for translation: {ss_err_trans * 1000} mm")
print(f"Single shooting error for rotation: {ss_err_rot * 180 / np.pi} degrees")

# --- compute contact forces ---
grf = []
cop = []
moments = []
m_contact = ([26], [26, 27, 28], [27, 28, 29])
rmse_grf = []
for p in range(len(phase_time) - 1):
    states=sol_previous.states[p]
    controls=sol_previous.controls[p]
    grf.append(compute_sum_forces(states["all"], controls["all"], ocp_previous.nlp[p]))
    cop.append(compute_cop(states["all"], controls["all"], ocp_previous.nlp[p], m_contact[p]))
    moments.append(compute_moments(states["all"], controls["all"], ocp_previous.nlp[p], m_contact[p]))
    rmse_grf.append(rmse(grf[p], grf_ref[p], grf=True))
plot_contact_forces(grf_ref, grf)

# --- compute marker positions ---
markers_pelvis = [0, 1, 2, 3]
markers_foot = [19, 20, 21, 22, 23, 24, 25]
rmse_markers = []
rmse_pelvis = []
rmse_foot=[]
for p in range(len(phase_time) - 1):
    q=sol_previous.states[p]["q"]
    m_pos = compute_markers_position(q, ocp_previous.nlp[p])
    rmse_markers.append(rmse(m_pos[:, :-4, :], markers_ref[p]))
    rmse_pelvis.append(rmse(m_pos[:, markers_pelvis, :], markers_ref[p][:, markers_pelvis, :]))
    rmse_foot.append(rmse(m_pos[:, markers_foot, :], markers_ref[p][:, markers_foot, :]))

print(f"error grf : {np.mean(rmse_grf)} N")
print(f"error all markers : {np.mean(rmse_markers) * 1e3} mm")
print(f"error pelvis : {np.mean(rmse_pelvis) * 1e3} mm")
print(f"error foot : {np.mean(rmse_foot) * 1e3} mm")