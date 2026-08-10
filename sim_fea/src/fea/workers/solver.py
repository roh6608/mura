import numpy as np
from loguru import logger
from shared.core.worker import Worker

from fea.core.config import SimConfig
from fea.core.types import FEMResult
from fea.geometry.mesh import MeshReader
from fea.geometry.renderer import Renderer
from fea.solver.fem import FEMSolver


class SolverWorker(Worker):
    """
    Reads each mesh file listed in the config and runs the FEA simulation on
    it. Exits once all files are processed or shutdown is requested.
    """

    def __init__(self, config: SimConfig) -> None:
        super().__init__()
        self.config = config

    def main(self) -> None:
        reader = MeshReader()
        renderer = Renderer()

        for file_path in self.config.files:
            if self.shutdown_requested:
                logger.info("Shutdown requested, aborting remaining simulations")
                break

            mesh = reader.load(file_path)
            if mesh is None:
                logger.error(f"Skipping {file_path}: mesh could not be loaded")
                continue

            solver = FEMSolver(
                mesh=mesh,
                youngs_modulus=self.config.materials.youngs_modulus,
                nu=self.config.materials.poissons_ratio,
                eigen_strains=self.config.eigen_strains,
                dirichlet_bcs=self.config.dirichlet_bcs,
            )
            solver.assemble_system()
            displacements = solver.solve()
            strains, stresses = solver.get_nodal_results()

            logger.success(
                f"{mesh.name}: max |u| = {np.abs(displacements).max():.3e}, "
                f"max |strain| = {np.abs(strains).max():.3e}, "
                f"max |stress| = {np.abs(stresses).max():.3e}"
            )

            result = FEMResult(
                mesh_name=mesh.name,
                nodes=mesh.nodes,
                elements=mesh.elements,
                displacements=displacements,
                strains=strains,
                stresses=stresses,
                group_map=mesh.group_map,
            )
            renderer.export(result, self.config.results_dir / f"{mesh.name}.vtu")
