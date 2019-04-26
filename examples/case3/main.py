import numpy as np

import sys; sys.path.insert(0, "../../src/")
from powerlaw import PowerLaw

sys.path.insert(0, "../common/")
import common

# In this test case we validate the MoLDD scheme for the Power Law model

# ------------------------------------------------------------------------------#

def test_mesh_size(solver):

    end_time = 1
    num_steps = 5
    time_step = end_time / num_steps

    # the flow problem
    param = {
        "tol": 1e-6,
        "k": 1,
        "aperture": 1e-2, "kf_t": 1e2, "kf_n": 1e2,
        "mass_weight": 1.0/time_step, # inverse of the time step
        "num_steps": num_steps,
        "L": 1,  # l-scheme constant
        "beta": 1,
        "r": 2.3,
    }

    # loop over the mesh sizes
    mesh_sizes = np.array([0.5, 0.125, 0.03125])
    num_iter = np.empty((mesh_sizes.size, num_steps))

    for idx, mesh_size in enumerate(mesh_sizes):
        # solve with MoLDD xor ItLDD scheme
        num_iter[idx, :]  = common.solve_(solver, mesh_size, param, PowerLaw)

    np.savetxt("powerlaw_mesh_size_" + solver + ".txt", num_iter, fmt="%d", delimiter=",")

# ------------------------------------------------------------------------------#

def test_time_step(solver):

    mesh_size = 0.125
    end_time = 1

    # the flow problem
    param = {
        "tol": 1e-6,
        "k": 1,
        "aperture": 1e-2, "kf_t": 1e2, "kf_n": 1e2,
        "L": 1,  # l-scheme constant
        "beta": 1,
        "r": 2.3,
    }

    # loop over the time steps
    num_steps = np.array([4, 8, 16])
    num_iter = np.zeros((num_steps.size, np.amax(num_steps)), dtype=np.int)

    for idx, num_step in enumerate(num_steps):
        time_step = end_time / num_step

        # consider the extra parameters
        param["mass_weight"] = 1.0/time_step
        param["num_steps"] = num_step

        # solve with MoLDD xor ItLDD scheme
        num_iter[idx, :num_step] = common.solve_(solver, mesh_size, param, PowerLaw)

    np.savetxt("powerlaw_time_step_" + solver + ".txt", num_iter, fmt="%d", delimiter=",")

# ------------------------------------------------------------------------------#

def test_parameters(solver):

    mesh_size = 0.125

    end_time = 1
    num_steps = 5
    time_step = end_time / num_steps

    # the flow problem
    param = {
        "tol": 1e-6,
        "k": 1,
        "aperture": 1e-2, "kf_t": 1e2, "kf_n": 1e2,
        "mass_weight": 1.0/time_step, # inverse of the time step
        "num_steps": num_steps,
        "L": 1,  # l-scheme constant
    }

    # change the value of beta
    betas = np.array([1e-1, 1, 10])
    param["r"] = 2.3
    num_iter_beta = np.empty((betas.size, num_steps), dtype=np.int)
    for idx, beta in enumerate(betas):
        # consider the parameters
        param["beta"] = beta

        # solve with MoLDD xor ItLDD scheme
        num_iter_beta[idx, :] = common.solve_(solver, mesh_size, param, PowerLaw)

    np.savetxt("powerlaw_beta_dependency_" + solver + ".txt", num_iter_beta, fmt="%d", delimiter=",")

    # change the value of r
    rs = np.array([4, 8, 16])
    param["beta"] = 1
    num_iter_r = np.empty((rs.size, num_steps), dtype=np.int)
    for idx, r in enumerate(rs):
        # consider the parameters
        param["r"] = r

        # solve with MoLDD xor ItLDD scheme
        num_iter_r[idx, :] = common.solve_(solver, mesh_size, param, PowerLaw)

    np.savetxt("powerlaw_r_dependency_" + solver + ".txt", num_iter_r, fmt="%d", delimiter=",")

# ------------------------------------------------------------------------------#

def test_L(solver):
    mesh_size = 0.125

    end_time = 1
    num_steps = 5
    time_step = end_time / num_steps

    # the flow problem
    param = {
        "tol": 1e-6,
        "k": 1,
        "aperture": 1e-2, "kf_t": 1e2, "kf_n": 1e2,
        "mass_weight": 1.0/time_step, # inverse of the time step
        "num_steps": num_steps,
        "beta": 1,
        "r": 2.3,
    }

    Ls = 0.025*np.arange(101)
    num_iter_L = np.empty((Ls.size, num_steps), dtype=np.int)
    for idx, L in enumerate(Ls):
        param["L"] = L

        # solve with MoLDD xor ItLDD scheme
        num_iter_L[idx, :] = common.solve_(solver, mesh_size, param, PowerLaw)

    np.savetxt("powerlaw_L_dependency_" + solver + ".txt", num_iter_L, fmt="%d", delimiter=",")

# ------------------------------------------------------------------------------#

def main(solver):

    h = 0.125

    end_time = 1
    num_steps = 5
    time_step = end_time / num_steps

    # the flow problem
    param = {
        "tol": 1e-6,
        "k": 1,
        "aperture": 1e-2, "kf_t": 1e2, "kf_n": 1e2,
        "mass_weight": 1.0/time_step, # inverse of the time step
        "num_steps": num_steps,
        "L": 1e3,  # l-scheme constant
        "beta": 1,  # non-linearity constant
        "r": 2.3,
    }

    # solve with MoLDD xor ItLDD scheme
    num_iter = common.solve_(solver, h, param, PowerLaw)

    print(num_iter)

# ------------------------------------------------------------------------------#

if __name__ == "__main__":
    # choose solving method: MoLDD or ItLDD
    solver = "Mono"
    # solver = "Iter"
    # test_mesh_size(solver)
    # test_time_step(solver)
    test_parameters(solver)
    # test_L(solver)
    # main(solver)
