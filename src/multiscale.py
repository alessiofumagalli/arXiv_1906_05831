import numpy as np
import scipy.sparse as sps

import porepy as pp


class Multiscale(object):
    def __init__(self, gb):
        # The grid bucket
        self.gb = gb
        # Higher dimensional grid, assumed to be 1
        self.g_h = self.gb.grids_of_dimension(self.gb.dim_max())[0]

        # Number of dofs for the higher dimensional grid
        self.dof_hn = np.empty(0, dtype=np.int)
        # Number of dofs for the co-dimensional grids
        self.dof_ln = np.empty(0, dtype=np.int)
        # Number of dofs for the higher dimensional interfaces
        self.dof_he = np.empty(0, dtype=np.int)
        # Number of dofs for the co-dimensional grid interfaces
        self.dof_le = np.empty(0, dtype=np.int)

        # Diffusion matrix of the higher dimensional domain
        self.A_h = None
        # Right-hand side of the higher dimensional domain
        self.b_h = None
        # Couple the 1 co-dimensional pressure to the mortar variables
        self.C_h = None
        # Realise the jump operator given the mortar variables
        self.C_l = None

        # LU factorization of the higher dimensional matrix
        self.LU = None
        # Non-homogeneous solution
        self.x_h = None
        # Bases
        self.bases = None

    # ------------------------------------------------------------------------------#

    def extract_dof(self, block_dof, full_dof):
        # count the dof for each block
        dof = np.cumsum(np.append(0, np.asarray(full_dof)))

        for pair, bi in block_dof.items():
            g = pair[0]
            dof_loc = np.arange(dof[bi], dof[bi+1])
            if isinstance(g, pp.Grid):
                # we are actually dealing with a grid
                if g is self.g_h:
                    # we consider the higher dimensional grid
                    # collect the dof separately
                    self.dof_hn = dof_loc
                else:
                    # we consider the lower dimensional grids
                    # collect the dof in a collective way
                    self.dof_ln = np.r_[self.dof_ln, dof_loc]
            else:
                # we are now dealing with an interface
                # check if we are on the higher dimensional grid interface
                g_h = g[0] if g[0].dim > g[1].dim else g[1]
                if g_h is self.g_h:
                    self.dof_he = np.r_[self.dof_he, dof_loc]
                else:
                    self.dof_le = np.r_[self.dof_le, dof_loc]

    # ------------------------------------------------------------------------------#

    def high_dim_matrices(self, A):

        # extract the blocks for the higher dimension
        dof = np.r_[self.dof_hn, self.dof_he]
        self.A_h = self.block(A, dof, dof)

        # Couple the 1 co-dimensional pressure to the mortar variables
        self.C_h = self.block(A, self.dof_he, self.dof_ln)

        # Realise the jump operator given the mortar variables
        self.C_l = self.block(A, self.dof_ln, self.dof_he)

    # ------------------------------------------------------------------------------#

    def high_dim_rhs(self, b):
        # extract the blocks for the higher dimension
        dof = np.r_[self.dof_hn, self.dof_he]
        self.b_h = b[dof]

    # ------------------------------------------------------------------------------#

    def compute_bases(self):

        # construct the rhs with all the active pressure dof in the 1 co-dimension
        if_p = np.zeros(self.C_h.shape[1], dtype=np.bool)
        pos = 0
        for g, _ in self.gb:
            if not (g is self.g_h):
                pos += g.num_faces
                # only the 1 co-dimensional grids are interesting
                if g.dim == self.g_h.dim - 1:
                    if_p[pos : pos + g.num_cells] = True
                pos += g.num_cells

        # compute the bases
        num_bases = np.sum(if_p)
        dof_bases = self.C_h.shape[0]
        self.bases = np.zeros((self.C_h.shape[1], self.C_h.shape[1]))

        # we solve many times the same problem, better to factorize the matrix
        self.LU = sps.linalg.factorized(self.A_h.tocsc())

        # solve to compute the ms bases functions for homogeneous boundary
        # conditions
        for dof_basis in np.where(if_p)[0]:
            rhs = np.zeros(if_p.size)
            rhs[dof_basis] = 1.0
            # project from the co-dimensional pressure to the Robin boundary
            # condition
            rhs = np.r_[[0] * self.dof_hn.size, -self.C_h * rhs]
            # compute the jump of the mortars
            self.bases[:, dof_basis] = self.C_l * self.LU(rhs)[-dof_bases:]

        # save the bases in a csr format
        self.bases = sps.csr_matrix(self.bases)

    # ------------------------------------------------------------------------------#

    def solve_non_homogeneous(self):
        # solve for non-zero boundary conditions and right-hand side
        dof_bases = self.C_h.shape[0]
        self.x_h = -self.C_l * self.LU(self.b_h)[-dof_bases:]

    # ------------------------------------------------------------------------------#

    def solve_low_dim(self, A, b, add_bases=True):
        # construct the problem in the fracture network
        A_l = np.empty((2, 2), dtype=np.object)
        b_l = np.empty(2, dtype=np.object)

        # the multiscale bases are thus inserted in the right block of the lower
        # dimensional problem
        A_l[0, 0] = self.block(A, self.dof_ln, self.dof_ln)
        # MoLDD - add bases to matrix, ItLDD - add bases to rhs
        if add_bases:
            A_l[0, 0] += self.bases
        # add to the right-hand side the non-homogeneous solution from the
        # higher dimensional problem
        b_l[0] = b[self.dof_ln] + self.x_h
        # in the case of > 1 co-dimensional problems
        A_l[0, 1] = self.block(A, self.dof_ln, self.dof_le)
        A_l[1, 0] = self.block(A, self.dof_le, self.dof_ln)
        A_l[1, 1] = self.block(A, self.dof_le, self.dof_le)
        b_l[1] = b[self.dof_le]

        # assemble and return
        A_l = sps.bmat(A_l, "csr")
        b_l = np.r_[tuple(b_l)]

        return sps.linalg.spsolve(A_l, b_l)

    # ------------------------------------------------------------------------------#

    def solve_high_dim(self, x_l):
        # compute the higher dimensional solution
        b = np.r_[[0] * self.dof_hn.size, -self.C_h * x_l[: self.C_h.shape[1]]]
        return self.LU(self.b_h + b)

    # ------------------------------------------------------------------------------#

    def concatenate(self, x_h=None, x_l=None):
        # save and export using standard algorithm
        x = np.zeros(self.b_h.size + x_l.size)
        if x_h is not None:
            x[: self.dof_hn.size] = x_h[: self.dof_hn.size]
        if x_l is not None:
            x[self.dof_hn.size : (self.dof_hn.size + self.dof_ln.size)] = x_l[: self.dof_ln.size]
        return x

    # ------------------------------------------------------------------------------#

    def block(self, A, I, J):
        return A.tocsr()[I, :].tocsc()[:, J]

    # ------------------------------------------------------------------------------#
