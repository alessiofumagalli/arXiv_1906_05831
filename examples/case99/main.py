import numpy as np
import porepy as pp

import sys; sys.path.insert(0, "../../src/")
from flow_discretization import Flow
from multiscale import Multiscale

# TODO: L-test, beta-test


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

    if g.dim == 2:
        labels[in_flow + out_flow] = "dir"
        bc_val[b_faces[in_flow]] = 0
        bc_val[b_faces[out_flow]] = 1
    else:
        labels[:] = "dir"
        bc_val[b_faces] = (b_face_centers[0, :] < 0.5).astype(np.float)

    return labels, bc_val


def main():

    h = 0.025
    tol = 1e-6
    mesh_args = {"mesh_size_frac": h}
    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 2}
    folder = "case1"

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
        "aperture": 1e-2, "kf_t": 1e2, "kf_n": 1e2,
        "mass_weight": 0, # stationary problem
        "L": 1, # l-scheme constant
        "beta": 1, # non-linearity constant
    }

    # declare the flow problem and the multiscale solver
    flow = Flow(gb, folder, tol)
    ms = Multiscale(gb)

    # set the data
    flow.data(param, bc_flag)

    # create the matrix for the Darcy problem
    A, _, b, block_dof, full_dof = flow.matrix_rhs()

    # extract the block dofs for the ms
    ms.extract_dof(block_dof, full_dof)

    # extract the higher domensional matrices
    ms.high_dim_matrices(A)
    ms.compute_bases()

    # extract the higher dimensional right-hand side and compute its contribution
    ms.high_dim_rhs(b)
    ms.solve_non_homogeneous()

    # assemble the problem in the lower dimensional problem
    x_l = ms.solve_low_dim(A, b)

    # solve the higher dimensional problem
    x_h = ms.solve_high_dim(x_l)

    # create the global vector
    x = ms.concatenate(x_h, x_l)

    # solve the problem
    flow.extract(x, block_dof, full_dof)
    flow.export()

if __name__ == "__main__":
    main()
