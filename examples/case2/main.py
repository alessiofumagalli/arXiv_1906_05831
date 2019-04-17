import numpy as np
import porepy as pp

import sys; sys.path.insert(0, "../../src/")
from algorithm import MoLDD
from forchheimer import Forchheimer

sys.path.insert(0, "../common/")
import common

# In this test case we validate the MoLDD scheme for the Forchheimer model

def main():

    tol = 1e-6
    end_time = 1
    num_steps = 5
    time_step = end_time / num_steps

    h = 0.125
    folder = "case2"

    gb = common.make_mesh(h)

    # the flow problem
    param = {
        "tol": tol,
        "k": 1,
        "aperture": 1e-2, "kf_t": 1e2, "kf_n": 1e2,
        "mass_weight": 1.0/time_step, # inverse of the time step
        "num_steps": num_steps,
        "L": 1e0,  # l-scheme constant
        "beta": 1e2,  # non-linearity constant
    }

    # declare the algorithm
    algo = MoLDD(gb, folder, Forchheimer, tol)

    # set the data
    algo.set_data(param, common.bc_flag)

    # data for the problem
    conv = 1e-5
    max_iter = 1e3

    # solve the problem
    num_iter = algo.solve(conv, max_iter)

    print(num_iter)

if __name__ == "__main__":
    main()
