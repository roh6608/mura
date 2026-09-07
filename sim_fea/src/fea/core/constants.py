"""Fixed invariants of the finite element model: element type, solver and tensor ordering."""

# element type 11 in gmsh corresponds to a tet10.
ELEMENT_TYPE = 11

AMG_MAX_COARSE = 500

# Component order of the symmetric strain and stress tensors in Voigt notation.
TENSOR_LABELS = ("xx", "yy", "zz", "xy", "yz", "zx")

# Node order of a VTK_QUADRATIC_TETRA expressed in gmsh tet10 indices: gmsh
# numbers the last two mid-edge nodes the other way round.
VTK_TET10_FROM_GMSH = [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]
