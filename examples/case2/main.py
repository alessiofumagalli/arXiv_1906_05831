import numpy as np
import porepy as pp

import sys; sys.path.insert(0, "../../src/")
from algorithm import MoLDD

def bc_flag(g, data, tol):
    b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
    b_face_centers = g.face_centers[:, b_faces]

    # define outflow type boundary conditions
    out_flow = b_face_centers[1] > 2 - tol

    # define inflow type boundary conditions
    in_flow = b_face_centers[1] < 0 + tol

    # define the labels and values for the boundary faces
    labels = np.array(["neu"] * b_faces.size)
    bc_val = np.zeros(g.num_faces)

    labels[:] = "dir" #

#    if g.dim == 2:
#        labels[in_flow + out_flow] = "dir"
#        bc_val[b_faces[in_flow]] = 0
#        bc_val[b_faces[out_flow]] = 1
#    else:
#        labels[:] = "dir"
#        bc_val[b_faces] = (b_face_centers[0, :] < 0.5).astype(np.float)

    return labels, bc_val

def main():

    h = 0.125
    tol = 1e-6
    time_step = 1e-1 # time step
    mesh_args = {"mesh_size_frac": h}
    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 2}
    folder = "case2"

    # Point coordinates, as a 2xn array
    p = np.array([[0, 1], [1, 1]])

    # Point connections as a 2 x num_frac arary
    e = np.array([[0], [1]])

    # Define a fracture network in 2d
    network_2d = pp.FractureNetwork2d(p, e, domain)

    # Generate a mixed-dimensional mesh
    gb = network_2d.mesh(mesh_args)
    #pp.plot_grid(gb, alpha=0, info="all")

    # the flow problem
    param = {
        "domain": gb.bounding_box(as_dict=True),
        "tol": tol,
        "k": 1,
        "aperture": 1e-2, "kf_t": 1e2, "kf_n": 1e8,
        "mass_weight": 1.0/time_step, # inverse of the time step
        "num_steps": 5,
        "L": 1,  # l-scheme constant
        "beta": 1,  # non-linearity constant
    }

    # declare the algorithm
    algo = MoLDD(gb, folder, tol)

    # set the data
    algo.data(param, bc_flag)

    # data for the problem
    conv = 1e-4
    max_iter = 100

    # solve the problem
    algo.solve(conv, max_iter)

if __name__ == "__main__":
    main()
