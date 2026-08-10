"""Example simulation configuration."""

from pathlib import Path

from fea.core.config import SimConfig
from fea.core.types import DirichletBC, EigenStrain, MaterialData, PhysicalGroup

config = SimConfig(
    materials=MaterialData(
        youngs_modulus=200.0e9,
        poissons_ratio=0.3,
    ),
    eigen_strains=[
        EigenStrain(
            group=PhysicalGroup.INCLUSION,
            exx=0.001,
            eyy=0.001,
            ezz=0.001,
            exy=0.0,
            eyz=0.0,
            ezx=0.0,
        ),
        EigenStrain(
            group=PhysicalGroup.MATRIX,
            exx=0.0,
            eyy=0.0,
            ezz=0.0,
            exy=0.0,
            eyz=0.0,
            ezx=0.0,
        ),
    ],
    dirichlet_bcs=[
        DirichletBC(group=PhysicalGroup.FIXED_NODES, ux=0.0, uy=0.0, uz=0.0),
    ],
    files=[
        Path("examples/model.msh"),
    ],
)
