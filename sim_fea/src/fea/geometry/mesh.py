"""Reading gmsh mesh files into the simulation's mesh dataclass."""

from pathlib import Path
from types import ModuleType

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from fea.core.constants import ELEMENT_TYPE
from fea.core.types import MeshData
from fea.utils.context_managers import gmsh_session


class MeshReader:
    """Reader for gmsh mesh files."""

    def __init__(self) -> None:
        """Create a reader holding no mesh."""

    def _gmsh_load_mesh_file(self, msh: ModuleType, file_path: str) -> None:
        """
        Open and load the gmsh file.

        :param msh: The initialised gmsh module from gmsh_session.
        :param file_path: File path to gmsh file.
        :type file_path: str
        """
        if not Path.exists(Path(file_path)):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        try:
            msh.open(file_path)
        except RuntimeError as e:
            logger.error(f"Internal Gmsh error: {e}")
            raise

    def _extract_nodes(self, msh: ModuleType) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
        node_tags, coords, _ = msh.model.mesh.getNodes()
        nodes = np.array(coords, dtype=np.float64).reshape(-1, 3)

        tag_mapper = np.zeros(int(np.max(node_tags)) + 1, dtype=np.int32)
        tag_mapper[node_tags] = np.arange(len(node_tags))

        return nodes, tag_mapper

    def load(self, file_path: Path) -> MeshData | None:
        """Load the gmsh file and extract it into a 'MeshData' dataclass.

        Work for the MeshWorker to complete: instantiates the gmsh context manager, opens the file, then extracts the
        mesh.

        :param filepath: filepath of gmsh file.
        :type filepath: str
        """
        try:
            with gmsh_session() as msh:
                self._gmsh_load_mesh_file(msh, str(file_path))

                node_tags, coords, _ = msh.model.mesh.getNodes()
                nodes = np.array(coords, dtype=np.float64).reshape(-1, 3)

                tag_mapper = np.zeros(int(np.max(node_tags)) + 1, dtype=np.int32)
                tag_mapper[node_tags] = np.arange(len(node_tags))

                node_groups = {}
                name_to_tag = {}

                for dim in range(4):
                    groups = msh.model.getPhysicalGroups(dim)
                    for _, g_tag in groups:
                        name = msh.model.getPhysicalName(dim, g_tag)
                        if not name:
                            continue

                        name_to_tag[name] = g_tag

                        _, group_node_tags = msh.model.mesh.getNodesForPhysicalGroup(dim, g_tag)

                        if group_node_tags.size > 0:
                            node_indices = tag_mapper[group_node_tags.astype(np.int32)]
                            node_groups[g_tag] = node_indices

                            dim_name = {0: "Point", 1: "Curve", 2: "Surface", 3: "Volume"}[dim]
                            logger.info(
                                f"Found Physical {dim_name}: '{name}' (ID: {g_tag}) with {len(node_indices)} nodes."
                            )

                all_elements = []
                all_tags = []
                entities = msh.model.getEntities(3)

                for _, entity_tag in entities:
                    phys_tags = msh.model.getPhysicalGroupsForEntity(3, entity_tag)
                    found_phys_id = phys_tags[0] if len(phys_tags) > 0 else 0

                    e_types, _, n_tags = msh.model.mesh.getElements(3, entity_tag)

                    for i in range(len(e_types)):
                        if e_types[i] == ELEMENT_TYPE:
                            raw_nodes = n_tags[i].astype(np.int32)
                            conn = tag_mapper[raw_nodes].reshape(-1, 10)

                            all_elements.append(conn)
                            all_tags.append(np.full(conn.shape[0], found_phys_id, dtype=np.int32))

                if not all_elements:
                    raise ValueError("No tetrahedral elements found in the mesh.")

                elements_final = np.vstack(all_elements)
                tags_final = np.concatenate(all_tags)

                logger.success(f"Mesh Loaded: {nodes.shape[0]} nodes, {elements_final.shape[0]} elements.")

                return MeshData(
                    nodes=nodes,
                    elements=elements_final,
                    element_tags=tags_final,
                    group_map=name_to_tag,
                    node_groups=node_groups,
                    name=Path(file_path).stem,
                )
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
        except (RuntimeError, AttributeError) as e:
            logger.error(f"GMSH Engine error on {file_path}: {e}")
