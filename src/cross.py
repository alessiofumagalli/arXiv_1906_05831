import numpy as np
import porepy as pp

from flow_discretization import Flow

# We consider the following generalised Darcy flow in the fractures
# \xi(u) + K^{-1} u + \nabla p = 0
# Where \xi follows the Cross model doi.org/10.1016/0095-8522(65)90022-X
# with \xi(u) = ( \alpha u ) / ( 1 + \beta |u|^{2-r} )
# For shear thinning fluids 1 < r < 2

class Cross(Flow):
    def __init__(self, gb, folder, tol):
        super().__init__(gb, folder, tol)

    # ------------------------------------------------------------------------------#

    def update_rhs_MoLDD(self):
        # retrieve problem specific data
        beta = self.data["beta"]
        alpha = self.data["beta"]
        r = self.data["r"]

        for g, d in self.gb:
            if g.dim == 1:
                # P0-projected velocity field
                P0u = d[self.P0_flux + "_old"]
                norm_u = np.linalg.norm(P0u, axis=0)

                # compute the non-linear term
                pow_u = np.power(norm_u, 2 - r)
                xi_u = np.divide(alpha, 1 + beta * pow_u)

                # non_linear and jacobian coefficient
                kf_inv = self.data["L"] - xi_u

                aperture = self.gb.node_props(g, pp.PARAMETERS)[self.model]["aperture"]
                kf = (1.0 / kf_inv / aperture) * np.ones(g.num_cells)

                # update permeability tensor
                perm = pp.SecondOrderTensor(1, kxx=kf, kyy=1, kzz=1)
                d[pp.PARAMETERS].modify_parameters("flow", "second_order_tensor", perm)

        # get updated flux inner product matrix
        return self.matrix_rhs()

    # ------------------------------------------------------------------------------#
