import os

import numpy as np
import matplotlib.pyplot as plt
import bioviz

from pendulum.ocp import prepare_ocp

root_path = "/".join(__file__.split("/")[:-1]) + "/"
model_path = root_path + "models/MassPoint_pendulum.bioMod"
np.random.seed(0)

ocp = prepare_ocp(biorbd_model_path=model_path)
sol = ocp.solve(show_online_optim=False)

q = np.hstack((sol.states[0]["q"], sol.states[1]["q"]))
qdot = np.hstack((sol.states[0]["qdot"], sol.states[1]["qdot"]))
u = np.hstack((sol.controls[0]["tau"], sol.controls[1]["tau"]))

time_vector = np.linspace(0, 10, 102)

fig = plt.figure(figsize=(6, 10))
plt.gcf().subplots_adjust(bottom=0.075, top=0.925, left=0.3, right=0.85, hspace=0.1)  # bottom=0.15 , wspace=0.35

axs_0 = plt.subplot(3, 1, 1)
fig.text(0.06, 0.795, "POSITIONS", va="center", ha="center", rotation="vertical", fontsize=18)
axs_0.plot(time_vector[:-1], q[0, :-1], "-", color="#1f77b4", label="Mass Position")
axs_0.axvline(x=5, color="k", linewidth=0.8, alpha=0.6, linestyle=(0, (5, 5)))
axs_0.tick_params(axis="x", labelcolor="w")
axs_0.set_ylabel("Mass position\n[m]", color="#1f77b4").set_fontsize(16)
axs_0.tick_params(axis="y", labelcolor="#1f77b4")

ax0 = axs_0.twinx()
ax0.plot(0, 0, "-", color="#1f77b4", label="Mass Position")
ax0.plot(time_vector[:-1], q[1, :-1], "-", color="#ff7f0e", label="Pendulum Position")
ax0.set_ylabel("Pendulum position\n[rad]", color="#ff7f0e").set_fontsize(16)
ax0.tick_params(axis="y", labelcolor="#ff7f0e")

axs_1 = plt.subplot(3, 1, 2)
fig.text(0.06, 0.5, "VELOCITIES", va="center", ha="center", rotation="vertical", fontsize=18)
axs_1.plot(time_vector[:-1], qdot[0, :-1], "-", color="#1f77b4", label="Mass Velocity")
(l1,) = axs_1.plot(
    np.array([5, 5]), np.array([-0.55, 0.5]), "-", linewidth=0.8, color="k", alpha=0.6, linestyle=(0, (5, 5))
)
axs_1.tick_params(axis="x", labelcolor="w")
axs_1.set_ylabel("Mass velocity\n[m/s]", color="#1f77b4").set_fontsize(16)
axs_1.tick_params(axis="y", labelcolor="#1f77b4")

ax1 = axs_1.twinx()
(lines1,) = ax1.plot(0, 0, "-", color="#1f77b4", label="Mass Velocity")
(lines2,) = ax1.plot(time_vector[:-1], qdot[1, :-1], "-", color="#ff7f0e", label="Pendulum Velocity")
ax1.set_ylabel("Pendulum velocity\n[rad/s]", color="#ff7f0e").set_fontsize(16)
ax1.tick_params(axis="y", labelcolor="#ff7f0e")

axs_1.legend(
    handles=[l1],
    labels=["Phase Transition"],
    bbox_to_anchor=(0.5, 2.3),
    loc="upper center",
    borderaxespad=0.0,
    frameon=False,
    fontsize=15,
)

axs_2 = plt.subplot(3, 1, 3)
fig.text(0.06, 0.215, "FORCES", va="center", ha="center", rotation="vertical", fontsize=18)
axs_2.step(time_vector[:-1], u[0, :-1], "-", color="#1f77b4", label="Mass Force")
axs_2.axvline(x=5, color="k", linewidth=0.8, alpha=0.6, linestyle=(0, (5, 5)))
axs_2.set_xlabel("Time [s]").set_fontsize(16)
axs_2.set_ylabel("Mass force actuation\n[N]", color="#1f77b4").set_fontsize(16)
axs_2.tick_params(axis="y", labelcolor="#1f77b4")

ax2 = axs_2.twinx()
ax2.plot(0, 0, "-", color="#1f77b4", label="Mass Force")
ax2.step(time_vector[:-1], -10 * q[0, :-1], "-", color="#2ca02c", label="Spring Force")
ax2.set_ylabel("Spring external force\n[N]", color="#2ca02c").set_fontsize(16)
ax2.tick_params(axis="y", labelcolor="#2ca02c")

try:
    os.mkdir(root_path + "figure")
except FileExistsError:
    pass
plt.savefig("figure/Mass_Pendulum_Fext.eps", format="eps")
plt.show()

print("RMS q_m - q*_m : ", np.std(q[0, 51:] - 0.5))

b = bioviz.Viz(model_path)
b.load_movement(q)
b.exec()
