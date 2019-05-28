import numpy as np
import porepy as pp

import sys; sys.path.insert(0, "../../src/")
from monolithic import MoLDD
from iterative import ItLDD

# ------------------------------------------------------------------------------#

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

# ------------------------------------------------------------------------------#

def make_mesh(mesh_size, plot=False):

    # define the domain
    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 2}

    # Point coordinates, as a 2xn array
    p = np.array([[0, 1], [1, 1]])

    # Point connections as a 2 x num_frac array
    e = np.array([[0], [1]])

    # Define a fracture network in 2d
    network_2d = pp.FractureNetwork2d(p, e, domain)

    # Generate a mixed-dimensional mesh
    gb = network_2d.mesh({"mesh_size_frac": mesh_size})

    if plot:
        pp.plot_grid(gb, alpha=0, info="all")

    return gb

# ------------------------------------------------------------------------------#

def solve_(solver, mesh_size, param, flow, conv=1e-5, max_iter=200):

    # create the grid bucket
    gb = make_mesh(mesh_size)

    # in case of the mortar
    if param.get("mortar_size", None):
        g_map = {}
        for g, _ in gb:
            if g.dim == 1:
                num_nodes = int(g.num_nodes * param["mortar_size"])
                g_map[g] = pp.refinement.remesh_1d(g, num_nodes)

        gb = pp.mortars.replace_grids_in_bucket(gb, g_map, {}, param["tol"])
        gb.assign_node_ordering()

    # declare the algorithm
    folder = "solution"
    if solver == "Iter":
        algo = ItLDD(gb, folder, flow, param["tol"])
    else:
        algo = MoLDD(gb, folder, flow, param["tol"])

    # set the data
    algo.set_data(param, bc_flag)

    # solve the problem
    return algo.solve(conv, max_iter)

# ------------------------------------------------------------------------------#

