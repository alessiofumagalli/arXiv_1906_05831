import numpy as np
from logger import logger
import porepy as pp

from multiscale import Multiscale

class MoLDD(object):

    def __init__(self, gb, folder, flow, tol):
        self.gb = gb
        # declare the flow problem and the multiscale solver
        self.flow = flow(gb, folder, tol)
        self.ms = Multiscale(gb)

        self.num_steps = 0

    # ------------------------------------------------------------------------------#

    def set_data(self, param, bc_flag):
        # set the data
        self.flow.set_data(param, bc_flag)
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

        # variable to save the number of iterations needed at each time step
        num_iter = np.zeros(self.num_steps, dtype=np.int)

        logger.info("Start the time loop with " + str(self.num_steps) + " steps")
        logger.add_tab()
        for n in np.arange(self.num_steps):
            logger.info("Time step " + str(n))
            # define the rhs for the current step
            rhs_time = b + M.dot(x)

            # extract the higher dimensional right-hand side and compute
            # its contribution
            logger.info("Extract the higher dimensional righ-hand side and compute its contribution")
            self.ms.high_dim_rhs(rhs_time)
            self.ms.solve_non_homogeneous()
            logger.info("done")

            i = 0
            err = np.inf
            logger.info("Start the non-linear loop with convergence " + str(conv) + " and " \
                        + str(max_iter) + "max iterations")
            logger.add_tab()
            while np.any(err > conv) and i < max_iter:
                logger.info("Perform iteration number " + str(i))

                # NOTE: we need to recompute only the lower dimensional matrices
                # for simplicity we do for everything otherwise a complex
                # mapping has to be coded. This point can be definitely
                # improved.
                logger.info("Re-compute the rhs due to the non-linear term")
                rhs = rhs_time + self.update_rhs()
                logger.info("done")

                logger.info("Re-compute the matrices due to the non-linear term")
                A = self.flow.update_matrix_MoLDD()[0]
                logger.info("done")

                # assemble the problem in the lower dimensional domain
                logger.info("Solve the lower dimensional problem")
                x_l = self.ms.solve_low_dim(A, rhs)
                logger.info("done")

                # distribute the variables to compute the error and to compute the
                # P0 projected flux, useful for the non-linear part
                logger.info("Compute the error")
                x = self.ms.concatenate(None, x_l)
                self.flow.extract(x, block_dof, full_dof)

                # compute the error to stop the non-linear loop
                err = self.compute_error()
                logger.info("done, error " + str(err))

                # save the variable with "_old" suffix
                self.save_old_variables()

                # increase the control loop variable
                i += 1
            logger.remove_tab()
            logger.info("done")

            # save the number of non linear iterations
            num_iter[n] = i

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

        return num_iter

    # ------------------------------------------------------------------------------#

    def update_rhs(self):
        # first update the stiffness matrix (fracture permeability)
        A, _, _, block_dof, full_dof = self.flow.update_rhs_MoLDD()

        # multiply A with fracture flux part of previous iteration vector
        x = np.zeros(A.shape[0])
        # count the dof for each block
        dof = np.cumsum(np.append(0, np.asarray(full_dof)))
        # find fracture flux dof
        for pair, bi in block_dof.items():
            g = pair[0]
            if isinstance(g, pp.Grid):
                # we are actually dealing with a grid
                if g.dim == 1:
                    d = self.gb.graph.node[g]
                    flux = d[self.flow.flux + "_old"]
                    # extract previous iteration flux to those dof
                    # consider in the A matrix only the block relative to the Hdiv mass
                    # matrix
                    dof_u = np.arange(dof[bi], dof[bi + 1])[:g.num_faces]
                    x[dof_u] = A[np.ix_(dof_u, dof_u)].dot(flux)

        # return the extra term for the rhs with previous iteration vector
        return x

    # ------------------------------------------------------------------------------#

    def save_old_variables(self):
        # extract the variable names
        pressure = self.flow.pressure
        flux = self.flow.flux
        P0_flux = self.flow.P0_flux

        for g, d in self.gb:
            d[pressure + "_old"] = d[pressure]
            d[flux + "_old"] = d[flux]
            d[P0_flux + "_old"] = d[P0_flux]

    # ------------------------------------------------------------------------------#

    def compute_error(self):
        # extract the variable names
        pressure = self.flow.pressure
        P0_flux = self.flow.P0_flux

        err = np.zeros(2)
        for g, d in self.gb:
            if g.dim < self.gb.dim_max():
                # compute the (relative) error for the pressure
                delta_p = np.power(d[pressure + "_old"] - d[pressure], 2)
                int_delta_p = g.cell_volumes.dot(delta_p)

                int_p = g.cell_volumes.dot(np.power(d[pressure + "_old"], 2))

                err[0] += int_delta_p / (int_p if int_p else 1)

                # compute the (relative) error for the reconstructed velocity
                delta_P0u = d[P0_flux + "_old"] - d[P0_flux]
                delta_P0u = np.einsum("ij,ij->j", delta_P0u, delta_P0u)
                int_delta_P0u = g.cell_volumes.dot(delta_P0u)

                P0u = np.einsum("ij,ij->j", d[P0_flux + "_old"], d[P0_flux + "_old"])
                int_P0u = g.cell_volumes.dot(P0u)

                err[1] += int_delta_P0u / (int_P0u if int_P0u else 1)

        return np.sqrt(err)

    # ------------------------------------------------------------------------------#


#class ItLDD(object):
