import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn

from somersault.ocp import prepare_ocp, prepare_ocp_quaternion


def run_ocp(is_quaternion):
    np.random.seed(0)

    if is_quaternion:
        ocp = prepare_ocp_quaternion(root_folder + "/models/JeChMesh_RootQuat.bioMod", final_time=1.5, n_shooting=100)
    else:
        ocp = prepare_ocp(root_folder + "/models/JeChMesh_8DoF.bioMod", final_time=1.5, n_shooting=100)

    sol = ocp.solve()
    return sol.states


root_folder = "/".join(__file__.split("/")[:-1])
q_euler = run_ocp(False)['q']
q_quaternion = run_ocp(True)['q']

time_vector = np.linspace(0, 100, 101)

seaborn.set_style("whitegrid")
seaborn.color_palette()

fig = plt.figure("Arm strategies")
plt.gcf().subplots_adjust(bottom=0.15, top=0.9, left=0.2, right=0.95, hspace=0.1)

ax0 = plt.subplot(2, 1, 1)
ax0.tick_params(axis="x", labelcolor="w")
ax0.set_ylabel("Right arm\nposition [°]", fontsize=15)
ax0.set_xlim(0, 100)
ax0.plot(time_vector, -q_euler[6, :] * 180 / np.pi, label="Euler angles")
ax0.plot(time_vector, -q_quaternion[6, :] * 180 / np.pi, label="quaternion")

ax1 = plt.subplot(2, 1, 2)
ax1.set_xlabel("Time [%]", fontsize=15)
ax1.set_ylabel("Left arm\nposition [°]", fontsize=15)
ax1.set_xlim(0, 100)
l1 = ax1.plot(time_vector, q_euler[7, :] * 180 / np.pi, label="Euler angles")
l2 = ax1.plot(time_vector, q_quaternion[7, :] * 180 / np.pi, label="quaternion")

ax1.legend(bbox_to_anchor=(0.5, 2.35), loc="upper center", borderaxespad=0.0, frameon=False, ncol=2, fontsize=15)

try:
    os.mkdir(root_folder + "figure")
except FileExistsError:
    pass
plt.savefig("figure/Twisting_armTech.eps", format="eps")
plt.show()
