import numpy as np
import biorbd
from bioptim import PlotType

from .ocp import Jumper5Phases


def com_height(x, nlp):
    q = nlp.mapping["q"].to_second.map(x[:7, :])
    com_func = biorbd.to_casadi_func("Compute_CoM", nlp.model.CoM, nlp.q)
    return np.array(com_func(q))[2]


def com_upward_velocity(x, nlp):
    q = nlp.mapping["q"].to_second.map(x[:7, :])
    qdot = nlp.mapping["q"].to_second.map(x[7:, :])
    com_dot_func = biorbd.to_casadi_func("Compute_CoM", nlp.model.CoMdot, nlp.q, nlp.qdot)
    return np.array(com_dot_func(q, qdot))[2]


def torque_bounds(x, min_or_max, nlp, minimal_tau=None):
    q = nlp.mapping["q"].to_second.map(x[:7, :])
    qdot = nlp.mapping["q"].to_second.map(x[7:, :])
    func = biorbd.to_casadi_func("TorqueMax", nlp.model.torqueMax, nlp.q, nlp.qdot)

    res = []
    for dof in [6, 7, 8, 9]:
        bound = []

        for i in range(len(x[0])):
            tmp = func(q[:, i], qdot[:, i])
            if minimal_tau and tmp[dof, min_or_max] < minimal_tau:
                bound.append(minimal_tau)
            else:
                bound.append(tmp[dof, min_or_max])
        res.append(np.array(bound))

    return np.array(res)


def add_jumper_plots(jumper: Jumper5Phases):
    for i in range(jumper.n_phases):
        nlp = jumper.ocp.nlp[i]
        # Plot Torque Bounds
        jumper.ocp.add_plot(
            "tau", lambda x, u, p: torque_bounds(x, 0, nlp), phase=i, plot_type=PlotType.STEP, color="g", linestyle="-."
        )
        jumper.ocp.add_plot(
            "tau",
            lambda x, u, p: -torque_bounds(x, 1, nlp),
            phase=i,
            plot_type=PlotType.STEP,
            color="g",
            linestyle="-.",
        )
        jumper.ocp.add_plot(
            "tau",
            lambda x, u, p: torque_bounds(x, 0, nlp, minimal_tau=jumper.tau_min),
            phase=i,
            plot_type=PlotType.STEP,
            color="g",
        )
        jumper.ocp.add_plot(
            "tau",
            lambda x, u, p: -torque_bounds(x, 1, nlp, minimal_tau=jumper.tau_min),
            phase=i,
            plot_type=PlotType.STEP,
            color="g",
        )
        jumper.ocp.add_plot(
            "tau", lambda x, u, p: np.zeros((4, len(x[0]))), phase=i, plot_type=PlotType.STEP, color="k"
        )

        # Plot center of mass pos and speed
        jumper.ocp.add_plot(
            "Center of mass height", lambda x, u, p: com_height(x, nlp), phase=i, plot_type=PlotType.PLOT
        )
        jumper.ocp.add_plot(
            "Center of mass upward velocity",
            lambda x, u, p: com_upward_velocity(x, nlp),
            phase=i,
            plot_type=PlotType.PLOT,
        )

        # Plot q and nb_qdot ranges
        jumper.ocp.add_plot(
            "q",
            lambda x, u, p: np.repeat(jumper.x_bounds[i].min[: jumper.n_q, 1][:, np.newaxis], x.shape[1], axis=1),
            phase=i,
            plot_type=PlotType.PLOT,
        )
        jumper.ocp.add_plot(
            "q",
            lambda x, u, p: np.repeat(jumper.x_bounds[i].max[: jumper.n_q, 1][:, np.newaxis], x.shape[1], axis=1),
            phase=i,
            plot_type=PlotType.PLOT,
        )
        jumper.ocp.add_plot(
            "qdot",
            lambda x, u, p: np.repeat(jumper.x_bounds[i].min[jumper.n_q :, 1][:, np.newaxis], x.shape[1], axis=1),
            phase=i,
            plot_type=PlotType.PLOT,
        )
        jumper.ocp.add_plot(
            "qdot",
            lambda x, u, p: np.repeat(jumper.x_bounds[i].max[jumper.n_q :, 1][:, np.newaxis], x.shape[1], axis=1),
            phase=i,
            plot_type=PlotType.PLOT,
        )
