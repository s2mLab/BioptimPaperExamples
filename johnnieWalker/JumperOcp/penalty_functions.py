import biorbd_casadi as biorbd
from casadi import MX, sum1
from bioptim import PenaltyNode, BiorbdInterface


def com_dot_z(pen_node: PenaltyNode):
    nlp = pen_node.nlp
    return BiorbdInterface.mx_to_cx("com_dot", nlp.model.CoMdot, nlp.states["q"], nlp.states["qdot"])


def marker_on_floor(pen_node: PenaltyNode, marker):
    nlp = pen_node.nlp
    return BiorbdInterface.mx_to_cx("toe_on_floor", nlp.model.marker, nlp.states["q"], marker)


def contact_force_continuity(pen_node: PenaltyNode, idx_pre, idx_post):
    final_contact_z = sum1(pen_node[0].nlp.contact_forces_func(pen_node[0].x[0], pen_node[0].u[0], pen_node[0].p)[idx_pre, :])
    if idx_post:
        starting_contact_z = sum1(pen_node[1].nlp.contact_forces_func(pen_node[1].x[0], pen_node[1].u[0], pen_node[1].p)[idx_post, :])
    else:
        starting_contact_z = MX.zeros(final_contact_z.shape)

    return final_contact_z - starting_contact_z
