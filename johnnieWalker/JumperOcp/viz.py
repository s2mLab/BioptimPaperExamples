import numpy as np
import biorbd_casadi as biorbd

from bioptim import PlotType


def plot_com(x, nlp):
    com_func = biorbd.to_casadi_func("CoMPlot", nlp.model.CoM, nlp.states["q"].mx, expand=False)
    com_dot_func = biorbd.to_casadi_func("Compute_CoM", nlp.model.CoMdot, nlp.states["q"].mx, nlp.states["qdot"].mx, expand=False)
    q = nlp.states["q"].mapping.to_second.map(x[nlp.states["q"].index, :])
    qdot = nlp.states["qdot"].mapping.to_second.map(x[nlp.states["qdot"].index, :])

    return np.concatenate((np.array(com_func(q)[2, :]), np.array(com_dot_func(q, qdot)[2, :])))


def add_custom_plots(ocp, jumper_ocp):
    nlp = ocp.nlp[0]

    ocp.add_plot(
        "tau_controls", lambda t, x, u, p: np.zeros((jumper_ocp.jumper.model.nbGeneralizedTorque(), len(x[0]))), plot_type=PlotType.STEP, color="k"
    )

    # Plot CoM pos and velocity
    ocp.add_plot("CoM", lambda t, x, u, p: plot_com(x, nlp), legend=["CoM", "CoM_dot"])

    # Plot q and nb_qdot ranges
    ocp.add_plot(
        "q_states",
        lambda t, x, u, p: np.repeat(jumper_ocp.x_bounds[0].min[: jumper_ocp.n_q, 1][:, np.newaxis], x.shape[1], axis=1),
        plot_type=PlotType.PLOT,
    )
    ocp.add_plot(
        "q_states",
        lambda t, x, u, p: np.repeat(jumper_ocp.x_bounds[0].max[: jumper_ocp.n_q, 1][:, np.newaxis], x.shape[1], axis=1),
        plot_type=PlotType.PLOT,
    )
    ocp.add_plot(
        "qdot_states",
        lambda t, x, u, p: np.repeat(jumper_ocp.x_bounds[0].min[jumper_ocp.n_q :, 1][:, np.newaxis], x.shape[1], axis=1),
        plot_type=PlotType.PLOT,
    )
    ocp.add_plot(
        "qdot_states",
        lambda t, x, u, p: np.repeat(jumper_ocp.x_bounds[0].max[jumper_ocp.n_q :, 1][:, np.newaxis], x.shape[1], axis=1),
        plot_type=PlotType.PLOT,
    )
