import numpy as np
import biorbd_casadi as biorbd

from bioptim import PlotType


def plot_com(x, nlp):
    com_func = biorbd.to_casadi_func("CoMPlot", nlp.model.CoM, nlp.states["q"].mx, expand=False)
    com_dot_func = biorbd.to_casadi_func("Compute_CoM", nlp.model.CoMdot, nlp.states["q"].mx, nlp.states["qdot"].mx, expand=False)
    q = nlp.states["q"].mapping.to_second.map(x[nlp.states["q"].index, :])
    qdot = nlp.states["qdot"].mapping.to_second.map(x[nlp.states["qdot"].index, :])

    return np.concatenate((np.array(com_func(q)[2, :]), np.array(com_dot_func(q, qdot)[2, :])))


def torque_bounds(x, min_or_max, nlp, minimal_tau=None):
    func = biorbd.to_casadi_func("torqueMaxPlot", nlp.model.torqueMax, nlp.states["q"].mx, nlp.states["qdot"].mx, expand=False)
    q = nlp.states["q"].mapping.to_second.map(x[nlp.states["q"].index, :])
    qdot = nlp.states["qdot"].mapping.to_second.map(x[nlp.states["qdot"].index, :])

    dof = [3, 5, 6, 7]
    res = np.array(func(q, qdot)[dof, ::2] if min_or_max == 0 else func(q, qdot)[dof, 1::2])
    if minimal_tau is not None:
        res[res < minimal_tau] = minimal_tau
    return res


def add_custom_plots(ocp, jumper_ocp):
    jumper = jumper_ocp.jumper
    for i, nlp in enumerate(ocp.nlp):

        # Plot Torque Bounds
        ocp.add_plot(
            "tau_controls", lambda t, x, u, p: torque_bounds(x, 0, nlp), phase=i, plot_type=PlotType.STEP, color="g", linestyle="-."
        )
        ocp.add_plot(
            "tau_controls",
            lambda t, x, u, p: -torque_bounds(x, 1, nlp),
            phase=i,
            plot_type=PlotType.STEP,
            color="g",
            linestyle="-.",
        )
        ocp.add_plot(
            "tau_controls",
            lambda t, x, u, p: torque_bounds(x, 0, nlp, minimal_tau=jumper.tau_min),
            phase=i,
            plot_type=PlotType.STEP,
            color="g",
        )
        ocp.add_plot(
            "tau_controls",
            lambda t, x, u, p: -torque_bounds(x, 1, nlp, minimal_tau=jumper.tau_min),
            phase=i,
            plot_type=PlotType.STEP,
            color="g",
        )
        ocp.add_plot(
            "tau_controls", lambda t, x, u, p: np.zeros((4, len(x[0]))), phase=i, plot_type=PlotType.STEP, color="k"
        )

        # Plot CoM pos and velocity
        ocp.add_plot("CoM", lambda t, x, u, p: plot_com(x, nlp), phase=i, legend=["CoM", "CoM_dot"])

        # Plot q and nb_qdot ranges
        ocp.add_plot(
            "q_states",
            lambda t, x, u, p: np.repeat(jumper_ocp.x_bounds[i].min[: jumper_ocp.n_q, 1][:, np.newaxis], x.shape[1], axis=1),
            phase=i,
            plot_type=PlotType.PLOT,
        )
        ocp.add_plot(
            "q_states",
            lambda t, x, u, p: np.repeat(jumper_ocp.x_bounds[i].max[: jumper_ocp.n_q, 1][:, np.newaxis], x.shape[1], axis=1),
            phase=i,
            plot_type=PlotType.PLOT,
        )
        ocp.add_plot(
            "qdot_states",
            lambda t, x, u, p: np.repeat(jumper_ocp.x_bounds[i].min[jumper_ocp.n_q :, 1][:, np.newaxis], x.shape[1], axis=1),
            phase=i,
            plot_type=PlotType.PLOT,
        )
        ocp.add_plot(
            "qdot_states",
            lambda t, x, u, p: np.repeat(jumper_ocp.x_bounds[i].max[jumper_ocp.n_q :, 1][:, np.newaxis], x.shape[1], axis=1),
            phase=i,
            plot_type=PlotType.PLOT,
        )
