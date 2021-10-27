
import numpy as np

from somersault.ocp import prepare_ocp, prepare_ocp_quaternion

import sys

if __name__ == "__main__":
    root_folder = "/".join(__file__.split("/")[:-1])

    f = open("SortieSolveur_QuatEuler.txt", "w+")
    f.write("Debut du code bitches!\n\n")
    f.close()

    for i in range(100):
        np.random.seed(i)


        sys.stdout = open("SortieSolveur_QuatEuler.txt", "a+")
        print(f"SEED = {i}\n\n")
        sys.stdout.close

        ocp = prepare_ocp_quaternion(root_folder + "/models/JeChMesh_RootQuat.bioMod", final_time=1.5, n_shooting=300)
        sol = ocp.solve()
        sys.stdout = open("SortieSolveur_QuatEuler.txt", "a+")
        print('Quaternion : \n')
        sol.print()
        sys.stdout.close

        ocp = prepare_ocp(root_folder + "/models/JeChMesh_8DoF.bioMod", final_time=1.5, n_shooting=300)
        sol = ocp.solve()
        sys.stdout = open("SortieSolveur_QuatEuler.txt", "a+")
        print('Euler : \n')
        sol.print()
        sys.stdout.close


    # sol.animate(patch_color=(0.4, 0.4, 0.4),
    #             background_color=(1.0, 1.0, 1.0),
    #             show_global_center_of_mass=False,
    #             show_segments_center_of_mass=False,
    #             show_global_ref_frame=False,
    #             show_local_ref_frame=False,
    #             show_markers=False,
    #             nb_frames=-1)

