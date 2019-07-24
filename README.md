# Robust linear domain decomposition schemes for reduced non-linear fracture flow models
Source code and examples for the paper<br>
"*Robust linear domain decomposition schemes for reduced non-linear fracture flow models*" by Elyes Ahmed, Alessio Fumagalli, Ana Budiša, Eirik Keilegavlen, Jan M. Nordbotten, A. Radu Florin. See [arXiv pre-print](https://arxiv.org/abs/1906.05831).

# Reproduce results from paper
Runscripts for all test cases of the work available [here](./examples).<br>
Note that you may have to revert to an older version of [PorePy](https://github.com/pmgbergen/porepy) to run the examples.

# Abstract
In this work, we consider compressible single-phase flow problems in a porous media containing a fracture. In the latter, a non-linear pressure-velocity relation is prescribed. Using a non-overlapping domain decomposition procedure, we reformulate the global problem into a non-linear interface problem. We then introduce two new algorithms that are able to efficiently handle the non-linearity and the coupling between the fracture and the matrix, both based on linearization by the so-called L-scheme. The first algorithm, named MoLDD, uses the L-scheme to resolve the non-linearity, requiring at each iteration to solve the dimensional coupling via a domain decomposition approach. The second algorithm, called ItLDD, uses a sequential approach in which the dimensional coupling is part of the linearization iterations. For both algorithms, the computations are reduced only to the fracture by pre-computing, in an offline phase, a multiscale flux basis (the linear Robin-to-Neumann co-dimensional map), that represent the flux exchange between the fracture and the matrix. We present extensive theoretical findings and in particular, the stability and the convergence of both schemes are obtained, where user given parameters are optimized to minimise the number of iterations. Examples on two important fracture models are computed with the library PorePy and agree with the developed theory.

# Citing
If you use this work in your research, we ask you to cite the following publication [arXiv:1906.05831 [math.NA]](https://arxiv.org/abs/1906.05831).

# PorePy version
If you want to run the code you need to install [PorePy](https://github.com/pmgbergen/porepy) and revert to commit 2378c6ee3d8f63e48ebdb8b2212c3989dfce1ecd <br>
Newer versions of PorePy may not be compatible with this repository.

# License
See [license](./LICENSE).
