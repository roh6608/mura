from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray


class PhysicalGroup(StrEnum):
    """
    Physical group names used in the mesh files. Members compare equal to
    their string values, so they can be used directly against gmsh group
    names.
    """

    INCLUSION = "Inclusion"
    MATRIX = "Matrix"
    FIXED_NODES = "FixedNodes"


@dataclass
class MeshData:
    """
    For storing mesh data.
    """

    nodes: NDArray[np.float64]
    elements: NDArray[np.int32]
    element_tags: NDArray[np.int32]
    group_map: dict[str, int]
    node_groups: dict[int, NDArray[np.int32]] = field(
        default_factory=dict[int, NDArray[np.int32]]
    )
    name: str = "unnamed_mesh"

    def get_nodes_by_group(self, phys_id: int) -> NDArray[np.int32]:
        """
        Returns the pre-calculated node indices for a specific physical group ID.
        """
        return self.node_groups.get(phys_id, np.array([], dtype=np.int32))

    def get_group_name(self, phys_id: int) -> str:
        """
        Reverse lookup to find a group name based on its ID.
        """
        for name, p_id in self.group_map.items():
            if p_id == phys_id:
                return name
        return "Unknown"


@dataclass
class FEMResult:
    """
    For storing FEM results.
    """

    mesh_name: str
    nodes: NDArray[np.float64]
    elements: NDArray[np.int32]
    displacements: NDArray[np.float64]
    strains: NDArray[np.float64] | None
    stresses: NDArray[np.float64] | None
    group_map: dict[str, int] | None


@dataclass
class MaterialData:
    """
    For storing material intrinsic properties.
    """

    youngs_modulus: float
    poissons_ratio: float


@dataclass(frozen=True)
class EigenStrain:
    """
    Prescribed eigenstrain components for a physical group (Voigt order).
    """

    group: PhysicalGroup
    exx: float
    eyy: float
    ezz: float
    exy: float
    eyz: float
    ezx: float


@dataclass(frozen=True)
class DirichletBC:
    """
    Prescribed displacement for a physical group. None means that axis is free.
    """

    group: PhysicalGroup
    ux: float | None
    uy: float | None
    uz: float | None
