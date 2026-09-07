"""Linear-elastic finite element solver for tet10 meshes carrying eigenstrain."""

import time

import numpy as np
import pyamg  # pyright: ignore[reportMissingTypeStubs]
import scipy.sparse as sp  # pyright: ignore[reportMissingTypeStubs]
import scipy.sparse.linalg as spla  # pyright: ignore[reportMissingTypeStubs]
from loguru import logger
from numpy.typing import NDArray

from fea.core.constants import AMG_MAX_COARSE
from fea.core.types import DirichletBC, EigenStrain, MeshData


class FEMSolver:
    """Linear-elastic finite element solver for tet10 meshes carrying eigenstrain."""

    def __init__(
        self,
        mesh: MeshData,
        youngs_modulus: float,
        nu: float,
        eigen_strains: list[EigenStrain],
        dirichlet_bcs: list[DirichletBC],
    ) -> None:
        """Store the mesh, material properties, eigenstrains and boundary conditions."""
        logger.info("Initialising FEM solver...")
        self.mesh = mesh
        self.youngs_modulus = youngs_modulus
        self.nu = nu
        self.num_nodes = mesh.nodes.shape[0]
        self.num_dof = self.num_nodes * 3

        self.nodes = np.asarray(mesh.nodes, dtype=np.float64)
        self.elements = np.asarray(mesh.elements, dtype=np.int32)
        self.element_tags = np.asarray(mesh.element_tags, dtype=np.int32)
        self.eigen_strains = eigen_strains
        self.dirichlet_bcs = dirichlet_bcs

        self.dmat = self._get_dmat_matrix()

        # weights for quadrature
        a = (5 + 3 * np.sqrt(5)) / 20
        b = (5 - np.sqrt(5)) / 20
        self.gp = np.array([[a, b, b], [b, a, b], [b, b, a], [b, b, b]])
        self.weights = np.array([0.25, 0.25, 0.25, 0.25]) / 6.0

        self.u_sol: NDArray[np.float64] | None = None
        self.kmat: sp.csr_matrix | None = None
        self.fvec: NDArray[np.float64] | None = None

        self.element_strains = None

    def _get_dmat_matrix(self) -> NDArray[np.float64]:
        mu = self.youngs_modulus / (2 * (1 + self.nu))
        lam = (self.youngs_modulus * self.nu) / ((1 + self.nu) * (1 - 2 * self.nu))
        dmat = np.zeros((6, 6), dtype=np.float64)
        dmat[0:3, 0:3] = lam
        np.fill_diagonal(dmat, 2 * mu + lam)
        dmat[3, 3] = mu
        dmat[4, 4] = mu
        dmat[5, 5] = mu
        return dmat

    def _get_quadratic_dn_dxi(self, xi: float, eta: float, zeta: float) -> NDArray[np.float64]:
        """Return the derivatives of 10-node Tetrahedron shape functions."""
        r = 1.0 - xi - eta - zeta

        return np.array(
            [
                [-(4 * r - 1), -(4 * r - 1), -(4 * r - 1)],  # N0
                [4 * xi - 1, 0, 0],  # N1
                [0, 4 * eta - 1, 0],  # N2
                [0, 0, 4 * zeta - 1],  # N3
                [4 * eta, 4 * xi, 0],  # N4
                [4 * (r - eta), -4 * eta, -4 * eta],  # N5
                [-4 * xi, 4 * (r - xi), -4 * xi],  # N6
                [0, 4 * zeta, 4 * eta],  # N7
                [-4 * zeta, -4 * zeta, 4 * (r - zeta)],  # N8
                [4 * zeta, 0, 4 * xi],  # N9
            ],
            dtype=np.float64,
        )

    def _construct_bmat_at_point(self, dn_dx: NDArray[np.float64]) -> NDArray[np.float64]:
        """Assembles B-matrix from global derivatives."""
        num_elems = dn_dx.shape[0]
        bmat = np.zeros((num_elems, 6, 30), dtype=np.float64)
        for i in range(10):
            col = 3 * i
            dn = dn_dx[:, i, :]
            bmat[:, 0, col] = dn[:, 0]
            bmat[:, 1, col + 1] = dn[:, 1]
            bmat[:, 2, col + 2] = dn[:, 2]
            bmat[:, 3, col + 1] = dn[:, 2]
            bmat[:, 3, col + 2] = dn[:, 1]
            bmat[:, 4, col] = dn[:, 2]
            bmat[:, 4, col + 2] = dn[:, 0]
            bmat[:, 5, col] = dn[:, 1]
            bmat[:, 5, col + 1] = dn[:, 0]
        return bmat

    def apply_dirichlet_bcs(self) -> None:
        """Apply the Dirichlet boundary conditions to the assembled system."""
        logger.info("Applying Dirichlet boundary conditions...")

        fixed_dofs: list[int] = []
        fixed_values: list[float] = []

        for bc in self.dirichlet_bcs:
            phys_id = self.mesh.group_map.get(bc.group)
            if phys_id is None:
                continue
            node_indices = self.mesh.get_nodes_by_group(phys_id)

            for node_idx in node_indices:
                for i, val in enumerate((bc.ux, bc.uy, bc.uz)):
                    if val is not None:
                        fixed_dofs.append(node_idx * 3 + i)
                        fixed_values.append(val)

        if not fixed_dofs:
            return

        dofs = np.asarray(fixed_dofs, dtype=np.int64)
        values = np.asarray(fixed_values, dtype=np.float64)

        u_fixed = np.zeros(self.num_dof, dtype=np.float64)
        u_fixed[dofs] = values
        self.fvec -= self.kmat @ u_fixed

        keep = np.ones(self.num_dof, dtype=np.float64)
        keep[dofs] = 0.0
        self.kmat = (sp.diags(keep) @ self.kmat @ sp.diags(keep) + sp.diags(1.0 - keep)).tocsr()
        self.fvec[dofs] = values

        logger.success("Boundary conditions applied.")

    def _constrain_dof(self, dof_idx: int, value: float) -> None:
        """Apply the Penalty Method to a specific Degree of Freedom."""
        penalty = 1e15
        self.kmat[dof_idx, dof_idx] = penalty
        self.fvec[dof_idx] = penalty * value

    def assemble_system(self) -> None:
        """Assemble the global stiffness matrix and the eigenstrain load vector."""
        logger.info("Assembling global system...")
        start = time.perf_counter()
        num_elems = self.elements.shape[0]

        self.element_strains = np.zeros((num_elems, 6), dtype=np.float64)

        for strain in self.eigen_strains:
            if strain.group in self.mesh.group_map:
                phys_id = self.mesh.group_map[strain.group]
                mask = self.element_tags == phys_id

                e_star = np.array(
                    [strain.exx, strain.eyy, strain.ezz, strain.exy, strain.eyz, strain.ezx], dtype=np.float64
                )

                self.element_strains[mask] = e_star

        sig_star_all = (self.dmat @ self.element_strains.T).T[:, :, None]

        ke_total = np.zeros((num_elems, 30, 30), dtype=np.float64)
        fe_total = np.zeros((num_elems, 30), dtype=np.float64)
        el_coords = self.nodes[self.elements]

        for pt in range(4):
            xi, eta, zeta = self.gp[pt]
            w = self.weights[pt]
            dn_dxi = self._get_quadratic_dn_dxi(xi, eta, zeta)

            jac = np.tensordot(el_coords, dn_dxi, axes=([1], [0]))
            det_j = np.linalg.det(jac)
            inv_j = np.linalg.inv(jac)
            dv = np.abs(det_j) * w

            dn_dx = np.matmul(dn_dxi[None, :, :], inv_j)
            bmat = self._construct_bmat_at_point(dn_dx)

            bt_d = np.matmul(bmat.transpose(0, 2, 1), self.dmat)
            ke_total += np.matmul(bt_d, bmat) * dv[:, None, None]

            fe_total += np.matmul(bmat.transpose(0, 2, 1), sig_star_all).squeeze() * dv[:, None]

        dof_indices = (self.elements[:, :, None] * 3 + np.arange(3)).reshape(num_elems, 30)
        rows = dof_indices[:, :, None].repeat(30, axis=2).flatten()
        cols = dof_indices[:, None, :].repeat(30, axis=1).flatten()

        self.kmat = sp.csr_matrix((ke_total.flatten(), (rows, cols)), shape=(self.num_dof, self.num_dof))
        self.fvec = np.zeros(self.num_dof, dtype=np.float64)
        np.add.at(self.fvec, dof_indices.flatten(), fe_total.flatten())

        logger.success(f"Assembly finished in {time.perf_counter() - start:.2f}s")

    def _get_rigid_body_modes(self) -> NDArray[np.float64]:
        """Return the six rigid body modes of the mesh.

        Three translations and three rotations.
        """
        x, y, z = self.nodes.T
        modes = np.zeros((self.num_dof, 6), dtype=np.float64)

        # translations
        modes[0::3, 0] = 1.0
        modes[1::3, 1] = 1.0
        modes[2::3, 2] = 1.0

        # rotations about x, y and z
        modes[1::3, 3] = -z
        modes[2::3, 3] = y
        modes[0::3, 4] = z
        modes[2::3, 4] = -x
        modes[0::3, 5] = -y
        modes[1::3, 5] = x

        return modes

    def _get_preconditioner(self) -> spla.LinearOperator:
        """Return a smoothed-aggregation AMG preconditioner."""
        start = time.perf_counter()
        ml = pyamg.smoothed_aggregation_solver(self.kmat, B=self._get_rigid_body_modes(), max_coarse=AMG_MAX_COARSE)
        logger.debug(f"AMG setup finished in {time.perf_counter() - start:.2f}s")
        return ml.aspreconditioner(cycle="V")

    def solve(self) -> NDArray[np.float64]:
        """Return the nodal displacements, applying the boundary conditions first."""
        logger.info("Applying boundary conditions and solving...")
        start = time.perf_counter()

        self.apply_dirichlet_bcs()

        m_mat = self._get_preconditioner()

        logger.info(f"Solving linear system with {self.num_dof} DOFs...")
        u_sol, info = spla.cg(self.kmat, self.fvec, M=m_mat, rtol=1e-5, maxiter=5000)

        if info > 0:
            logger.warning(f"CG did not converge after {info} iterations.")
        elif info < 0:
            logger.error("CG breakdown: illegal input or breakdown occurred.")

        self.u_sol = u_sol.astype(np.float64)
        logger.success(f"CPU Solve finished in {time.perf_counter() - start:.2f}s")

        return self.u_sol.reshape(-1, 3)

    def get_nodal_results(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Calculate nodal strains and stresses using weighted averaging."""
        logger.info("Calculating nodal strains/stresses...")
        start = time.perf_counter()

        num_elems = self.elements.shape[0]

        nodal_strain_sum = np.zeros((self.num_nodes, 6), dtype=np.float64)
        nodal_stress_sum = np.zeros((self.num_nodes, 6), dtype=np.float64)
        nodal_count = np.zeros(self.num_nodes, dtype=np.float64)

        el_coords = self.nodes[self.elements]
        u_elem = self.u_sol[self.elements[:, :, None] * 3 + np.arange(3)].reshape(num_elems, 30, 1)

        for pt in range(4):
            xi, eta, zeta = self.gp[pt]
            dn_dxi = self._get_quadratic_dn_dxi(xi, eta, zeta)

            jac = np.tensordot(el_coords, dn_dxi, axes=([1], [0]))
            dn_dx = np.matmul(dn_dxi[None, :, :], np.linalg.inv(jac))
            bmat = self._construct_bmat_at_point(dn_dx)

            gp_strain_total = np.matmul(bmat, u_elem).squeeze()

            if self.element_strains is None:
                gp_elastic_strain = gp_strain_total
            else:
                gp_elastic_strain = gp_strain_total - self.element_strains

            gp_stress = (self.dmat @ gp_elastic_strain.T).T

            for i in range(10):
                node_indices = self.elements[:, i]
                np.add.at(nodal_strain_sum, node_indices, gp_elastic_strain)
                np.add.at(nodal_stress_sum, node_indices, gp_stress)
                np.add.at(nodal_count, node_indices, 1.0)

        nodal_strains = nodal_strain_sum / nodal_count[:, None]
        nodal_stresses = nodal_stress_sum / nodal_count[:, None]

        logger.success(f"Nodal results ready in {time.perf_counter() - start:.2f}s")
        return nodal_strains, nodal_stresses
