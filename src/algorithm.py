import numpy as np
from logger import logger

from flow_discretization import Flow
from multiscale import Multiscale

class MoLDD(object):

    def __init__(self, gb, folder, tol):
        self.gb = gb
        # declare the flow problem and the multiscale solver
        self.flow = Flow(gb, folder, tol)
        self.ms = Multiscale(gb)

        self.num_steps = 0

    # ------------------------------------------------------------------------------#

    def data(self, param, bc_flag):
        # set the data
        self.flow.data(param, bc_flag)
        self.num_steps = param["num_steps"]

    # ------------------------------------------------------------------------------#

    def solve(self, conv, max_iter):

        # create the matrix for the Darcy problem
        logger.info("Create the matrices and rhs for the problem")
        A, M, b, block_dof, full_dof = self.flow.matrix_rhs()
        logger.info("done")

        # extract the block dofs for the ms
        logger.info("Extract the block dofs for the multiscale")
        self.ms.extract_dof(block_dof, full_dof)
        logger.info("done")

        # extract the higher dimensional matrices and compute the ms bases
        logger.info("Extract the higher dimensional matrices and compute the multiscale bases")
        self.ms.high_dim_matrices(A)
        self.ms.compute_bases()
        logger.info("done")

        # solution vector
        x = np.zeros(b.size)

        # save the variable with "_old" suffix
        self.flow.extract(x, block_dof, full_dof)
        self.save_old_variables()

        logger.info("Start the time loop with " + str(self.num_steps) + " steps")
        logger.add_tab()
        for n in np.arange(self.num_steps):
            logger.info("Time step " + str(n))
            # define the rhs for the current step
            rhs = b + M.dot(x)

            # extract the higher dimensional right-hand side and compute
            # its contribution
            logger.info("Extract the higher dimensional righ-hand side and compute its contribution")
            self.ms.high_dim_rhs(rhs)
            self.ms.solve_non_homogeneous()
            logger.info("done")

            i = 0
            err = np.inf
            logger.info("Start the non-linear loop with convergence " + str(conv) + " and " \
                        + str(max_iter) + "max iterations")
            logger.add_tab()
            while np.any(err > conv) and i < max_iter:
                logger.info("Perform iteration number " )

                # NOTE: we need to recompute only the lower dimensional matrices
                # for simplicity we do for everything otherwise a complex mapping
                # has to be coded. This point can be definitely improved.
                logger.info("Re-compute the matrices due to the non-linear term")
                A = self.flow.matrix_rhs()[0]
                logger.info("done")

                # assemble the problem in the lower dimensional problem
                logger.info("Solve the lower dimensional problem")
                x_l = self.ms.solve_low_dim(A, rhs)
                logger.info("done")

                # distribute the variables to compute the error
                logger.info("Compute the error")
                x = self.ms.concatenate(None, x_l)
                self.flow.extract(x, block_dof, full_dof)

                # save the variable with "_old" suffix
                self.save_old_variables()

                # compute the error to stop the non-linear loop
                err = self.compute_error()
                logger.info("done, error " + str(err))

                # increase the control loop variable
                i += 1
            logger.remove_tab()
            logger.info("done")

            # solve the higher dimensional problem
            logger.info("Solve the high dimensional problem")
            x_h = self.ms.solve_high_dim(x_l)
            logger.info("done")

            # create the full solution vector
            x = self.ms.concatenate(x_h, x_l)
            self.flow.extract(x, block_dof, full_dof)

            # save the variable with "_old" suffix
            self.save_old_variables()

            # solve the problem
            self.flow.export(time_step=n)

        # save the pvd
        self.flow.export_pvd(np.arange(self.num_steps))

        logger.remove_tab()
        logger.info("done")

    # ------------------------------------------------------------------------------#

    def save_old_variables(self):
        # extract the variable names
        pressure = self.flow.pressure
        flux = self.flow.flux

        for g, d in self.gb:
            d[pressure + "_old"] = d[pressure]
            d[flux + "_old"] = d[flux]

    # ------------------------------------------------------------------------------#

    def compute_error(self):
        # compute the error
        # TODO the denominator

        # extract the variable names
        pressure = self.flow.pressure
        flux = self.flow.flux

        err = np.zeros(2)
        for g, d in self.gb:
            if g.dim < self.gb.dim_max():
                err[0] += np.linalg.norm(d[pressure + "_old"] - d[pressure]) ** 2
                err[1] += np.linalg.norm(d[flux + "_old"] - d[flux]) ** 2

        return np.sqrt(err)

    # ------------------------------------------------------------------------------#


#class ItLDD(object):
