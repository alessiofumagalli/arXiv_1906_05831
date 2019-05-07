import numpy as np

import sys; sys.path.insert(0, "../../src/")
from cross import Cross

sys.path.insert(0, "../common/")
import common

# In this test case we validate the MoLDD scheme for the Cross model

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
        "mass_weight": 1.0/time_step,  # inverse of the time step
        "num_steps": num_steps,
        "L": 1,  # l-scheme constant
        "L_p": 1e3,  # inner l-scheme for iterative solver
        "alpha": 1,
        "zeta": 1,  # non-linearity constant
        "r": 1.5,
    }

    # loop over the mesh sizes
    mesh_sizes = np.array([0.5, 0.125, 0.03125])
    num_iter = np.empty((mesh_sizes.size, num_steps))

    for idx, mesh_size in enumerate(mesh_sizes):
        # solve with MoLDD xor ItLDD scheme
        num_iter[idx, :] = common.solve_(solver, mesh_size, param, Cross)

    np.savetxt("cross_mesh_size_" + solver + ".txt", num_iter, fmt="%d", delimiter=",")

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
        "L_p": 1e3,  # inner l-scheme for iterative solver
        "alpha": 1,
        "zeta": 1,  # non-linearity constant
        "r": 1.5,
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
        num_iter[idx, :num_step] = common.solve_(solver, mesh_size, param, Cross)

    np.savetxt("cross_time_step_" + solver + ".txt", num_iter, fmt="%d", delimiter=",")

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
        "mass_weight": 1.0/time_step,  # inverse of the time step
        "num_steps": num_steps,
        "L": 1,  # l-scheme constant
        "L_p": 1e3,  # inner l-scheme for iterative solver
    }

    # change the value of alpha
    alphas = np.array([1e-1, 1., 2.5])
    param["zeta"] = 1
    param["r"] = 1.5
    num_iter_alpha = np.empty((alphas.size, num_steps), dtype=np.int)
    for idx, alpha in enumerate(alphas):
        # consider the parameters
        param["alpha"] = alpha

        # solve with MoLDD xor ItLDD scheme
        num_iter_alpha[idx, :] = common.solve_(solver, mesh_size, param, Cross)

    np.savetxt("cross_alpha_dependency_" + solver + ".txt", num_iter_alpha, fmt="%d", delimiter=",")


    # change the value of zeta
    zetas = np.array([1., 10., 1e2])
    param["alpha"] = 1
    param["r"] = 1.5
    num_iter_zeta = np.empty((zetas.size, num_steps), dtype=np.int)
    for idx, zeta in enumerate(zetas):
        # consider the parameters
        param["zeta"] = zeta

        # solve with MoLDD xor ItLDD scheme
        num_iter_zeta[idx, :] = common.solve_(solver, mesh_size, param, Cross)

    np.savetxt("cross_zeta_dependency_" + solver + ".txt", num_iter_zeta, fmt="%d", delimiter=",")

    # change the value of r
    rs = np.array([1., 1.5, 4.5])
    param["alpha"] = 1
    param["zeta"] = 1
    num_iter_r = np.empty((rs.size, num_steps), dtype=np.int)
    for idx, r in enumerate(rs):
        # consider the parameters
        param["r"] = r

        # solve with MoLDD xor ItLDD scheme
        num_iter_r[idx, :] = common.solve_(solver, mesh_size, param, Cross)

    np.savetxt("cross_r_dependency_" + solver + ".txt", num_iter_r, fmt="%d", delimiter=",")

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
        "mass_weight": 1.0/time_step,  # inverse of the time step
        "num_steps": num_steps,
        "L_p": 1e3,  # inner l-scheme for iterative solver
        "alpha": 1,
        "zeta": 1,  # non-linearity constant
        "r": 1.5,
    }

    Ls = 0.025*np.arange(101)
    num_iter_L = np.empty((Ls.size, num_steps), dtype=np.int)
    for idx, L in enumerate(Ls):
        param["L"] = L

        # solve with MoLDD xor ItLDD scheme
        num_iter_L[idx, :] = common.solve_(solver, mesh_size, param, Cross)

    np.savetxt("cross_L_dependency_" + solver + ".txt", num_iter_L, fmt="%d", delimiter=",")

# ------------------------------------------------------------------------------#

def test_L_Lp(solver):
    mesh_size = 0.125

    end_time = 0.2
    num_steps = 1
    time_step = end_time / num_steps

    # the flow problem
    param = {
        "tol": 1e-6,
        "k": 1,
        "aperture": 1e-2, "kf_t": 1e2, "kf_n": 1e2,
        "mass_weight": 1.0 / time_step,  # inverse of the time step
        "num_steps": num_steps,
        "alpha": 1,
        "zeta": 1,  # non-linearity constant
        "r": 1.5,
    }

    Ls = 0.25*np.arange(11)
    Lps = np.power(10, np.arange(2.2, 4.3, 0.2))

    num_iter_L = np.empty((Ls.size, Lps.size), dtype=np.int)
    for row, L in enumerate(Ls):
        param["L"] = L

        for col, Lp in enumerate(Lps):
            param["L_p"] = Lp

            # solve with MoLDD xor ItLDD scheme
            num_iter_L[row, col] = common.solve_(solver, mesh_size, param, Cross)

    np.savetxt("cross_L_Lp_dependency_" + solver + ".txt", num_iter_L, fmt="%d", delimiter=",")

# ------------------------------------------------------------------------------#

def main(solver):

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
        "L": 1e2,  # l-scheme constant
        "L_p": 1e3,  # inner l-scheme for iterative solver
        "alpha": 1,
        "zeta": 1,  # non-linearity constant
        "r": 1.5,
    }

    # solve with MoLDD xor ItLDD scheme
    num_iter = common.solve_(solver, mesh_size, param, Cross)

    print(num_iter)

# ------------------------------------------------------------------------------#

if __name__ == "__main__":
    # choose solving method: MoLDD or ItLDD
    # solver = "Mono"
    solver = "Iter"
    # test_mesh_size(solver)
    # test_time_step(solver)
    # test_parameters(solver)
    # test_L(solver)
    test_L_Lp(solver)
    # main(solver)
