"""
This is an example on how to use quaternion to represent the orientation of the root of the model.
The avatar must complete one somersault rotation while maximizing the twist rotation.
"""
import numpy as np
from bioptim import Solver
from somersault.ocp import prepare_ocp_quaternion, prepare_ocp


if __name__ == "__main__":
    root_folder = "/".join(__file__.split("/")[:-1])
    is_quaternion = False
    is_collocation = False

    if is_quaternion:
        ocp = prepare_ocp_quaternion(root_folder + "/models/JeChMesh_RootQuat.bioMod", final_time=1.5, n_shooting=100, is_collocation=is_collocation)
    else:
        ocp = prepare_ocp(root_folder + "/models/JeChMesh_8DoF.bioMod", final_time=1.5, n_shooting=100, is_collocation=is_collocation)

    solver = Solver.IPOPT()
    solver.set_convergence_tolerance(1e-15)
    solver.set_acceptable_constr_viol_tol(1e-15)
    solver.set_maximum_iterations(1000)
    sol = ocp.solve(solver)
    sol.animate(nb_frames=-1)

    sol.animate(# patch_color=(0.4, 0.4, 0.4),
                # background_color=(1.0, 1.0, 1.0),
                # show_global_center_of_mass=False,
                # show_segments_center_of_mass=False,
                # show_global_ref_frame=False,
                # show_local_ref_frame=False,
                # show_markers=False,
                nb_frames=-1)

    q = sol.states["q"]
    qdot = sol.states["qdot"]
    xyzw = np.vstack((q[3:6, :], q[8, :]))
    norm = np.linalg.norm(xyzw, axis=0)
    print(norm)


    def states_to_euler_rate(q, qdot):
        # maximizing Lagrange twist velocity (indeterminate of quaternion to Euler of 2*pi*n)

        def body_vel_to_euler_rate(w, e):
            # xyz convention
            _ = e[0]
            th = e[1]
            ps = e[2]
            wx = w[0]
            wy = w[1]
            wz = w[2]
            dph = np.cos(ps) / np.cos(th) * wx - np.sin(ps) / np.cos(th) * wy
            dth = np.sin(ps) * wx + np.cos(ps) * wy
            dps = -np.cos(ps) * np.sin(th) / np.cos(th) * wx + np.sin(th) * np.sin(ps) / np.cos(th) * wy + wz
            return np.array([dph, dth, dps])

        N = len(q[0, :])
        euler_rates = np.zeros((3, N))
        for i in range(N):
            quaternion_cas = np.array([q[8, i], q[3, i], q[4, i], q[5, i]])
            quaternion_cas /= np.linalg.norm(quaternion_cas)
            quaternion = biorbd.Quaternion(quaternion_cas[0], quaternion_cas[1], quaternion_cas[2], quaternion_cas[3])

            omega = qdot[3:6, i]
            euler = biorbd.Rotation.toEulerAngles(biorbd.Quaternion.toMatrix(quaternion), "xyz").to_array()
            euler_rates[:, i] = body_vel_to_euler_rate(omega, euler)
        return euler_rates


    euler_rates = states_to_euler_rate(q, qdot)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(euler_rates[0, :], label='vitesse x')
    plt.plot(euler_rates[1, :], label='vitesse y')
    plt.plot(euler_rates[2, :], label='vitesse z')
    plt.legend()
    plt.show()

    dt = sol.parameters['time']/N
    np.sum(euler_rates[2, :] * dt)

    import scipy
    scipy.integrate.quad(euler_rates[2, :])

    import bioviz
    b = bioviz.Viz(root_folder + "/models/JeChMesh_RootQuat.bioMod", # RootQuat # 8DoF
                            show_global_center_of_mass=False,
                            show_segments_center_of_mass=False,
                            show_global_ref_frame=False,
                            show_local_ref_frame=False,
                            show_markers=False,
                            background_color=(1.0, 1.0, 1.0),
                            patch_color=(0.4, 0.4, 0.4))
    b.load_movement(q)
    b.exec()



    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(qdot[3, :], label='vitesse x')
    plt.plot(qdot[4, :], label='vitesse y')
    plt.plot(qdot[5, :], label='vitesse z')
    plt.legend()
    plt.show()

    N = len(q[0, :])
    dt = sol.parameters['time']/N
    np.sum(qdot[5, :] * dt)