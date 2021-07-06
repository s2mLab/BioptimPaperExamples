import numpy as np
import biorbd
from bioptim import PlotType

from .ocp import Jumper5Phases


def com_height(x, nlp):
    com_func = biorbd.to_casadi_func("CoMPlot", nlp.model.CoM, nlp.states["q"].mx)
    q = nlp.states["q"].mapping.to_second.map(x[nlp.states["q"].index, :])
    return np.array(com_func(q)[2, :])


def com_upward_velocity(x, nlp):
    com_dot_func = biorbd.to_casadi_func("Compute_CoM", nlp.model.CoMdot, nlp.states["q"].mx, nlp.states["qdot"].mx)
    q = nlp.states["q"].mapping.to_second.map(x[nlp.states["q"].index, :])
    qdot = nlp.states["qdot"].mapping.to_second.map(x[nlp.states["qdot"].index, :])
    return np.array(com_dot_func(q, qdot)[2, :])


def torque_bounds(x, min_or_max, nlp, minimal_tau=None):
    func = biorbd.to_casadi_func("torqueMaxPlot", nlp.model.torqueMax, nlp.states["q"].mx, nlp.states["qdot"].mx)
    q = nlp.states["q"].mapping.to_second.map(x[nlp.states["q"].index, :])
    qdot = nlp.states["qdot"].mapping.to_second.map(x[nlp.states["qdot"].index, :])

    dof = [3, 5, 6, 7]
    res = np.array(func(q, qdot)[dof, ::2] if min_or_max == 0 else func(q, qdot)[dof, 1::2])
    if minimal_tau is not None:
        res[res < minimal_tau] = minimal_tau
    return res


def add_jumper_plots(jumper: Jumper5Phases):
    for i in range(jumper.n_phases):
        nlp = jumper.ocp.nlp[i]
        # Plot Torque Bounds
        jumper.ocp.add_plot(
            "tau_controls", lambda x, u, p: torque_bounds(x, 0, nlp), phase=i, plot_type=PlotType.STEP, color="g", linestyle="-."
        )
        jumper.ocp.add_plot(
            "tau_controls",
            lambda x, u, p: -torque_bounds(x, 1, nlp),
            phase=i,
            plot_type=PlotType.STEP,
            color="g",
            linestyle="-.",
        )
        jumper.ocp.add_plot(
            "tau_controls",
            lambda x, u, p: torque_bounds(x, 0, nlp, minimal_tau=jumper.tau_min),
            phase=i,
            plot_type=PlotType.STEP,
            color="g",
        )
        jumper.ocp.add_plot(
            "tau_controls",
            lambda x, u, p: -torque_bounds(x, 1, nlp, minimal_tau=jumper.tau_min),
            phase=i,
            plot_type=PlotType.STEP,
            color="g",
        )
        jumper.ocp.add_plot(
            "tau_controls", lambda x, u, p: np.zeros((4, len(x[0]))), phase=i, plot_type=PlotType.STEP, color="k"
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
            "q_states",
            lambda x, u, p: np.repeat(jumper.x_bounds[i].min[: jumper.n_q, 1][:, np.newaxis], x.shape[1], axis=1),
            phase=i,
            plot_type=PlotType.PLOT,
        )
        jumper.ocp.add_plot(
            "q_states",
            lambda x, u, p: np.repeat(jumper.x_bounds[i].max[: jumper.n_q, 1][:, np.newaxis], x.shape[1], axis=1),
            phase=i,
            plot_type=PlotType.PLOT,
        )
        jumper.ocp.add_plot(
            "qdot_states",
            lambda x, u, p: np.repeat(jumper.x_bounds[i].min[jumper.n_q :, 1][:, np.newaxis], x.shape[1], axis=1),
            phase=i,
            plot_type=PlotType.PLOT,
        )
        jumper.ocp.add_plot(
            "qdot_states",
            lambda x, u, p: np.repeat(jumper.x_bounds[i].max[jumper.n_q :, 1][:, np.newaxis], x.shape[1], axis=1),
            phase=i,
            plot_type=PlotType.PLOT,
        )
