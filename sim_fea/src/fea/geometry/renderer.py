"""Exporting solver results to VTK unstructured grids."""

from pathlib import Path

import numpy as np
import vtk  # pyright: ignore[reportMissingTypeStubs]
from loguru import logger
from numpy.typing import NDArray
from vtk.util import numpy_support  # pyright: ignore[reportMissingTypeStubs]

from fea.core.constants import TENSOR_LABELS, VTK_TET10_FROM_GMSH
from fea.core.types import FEMResult


class Renderer:
    """Writer that exports solver results to VTK unstructured grids."""

    def __init__(self) -> None:
        """Create a renderer holding no grid."""

    def _to_vtk_array(self, data: NDArray[np.float64], name: str) -> vtk.vtkDataArray:
        """Convert a numpy array to a named VTK array.

        The data is copied, so the VTK array does not alias the numpy buffer.
        """
        vtk_array = numpy_support.numpy_to_vtk(np.ascontiguousarray(data), deep=1)
        vtk_array.SetName(name)
        return vtk_array

    def _build_grid(self, result: FEMResult) -> vtk.vtkUnstructuredGrid:
        """Build the unstructured grid of tet10 cells with the nodal results attached as point data."""
        num_elems = result.elements.shape[0]

        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(self._to_vtk_array(result.nodes, "Points"))

        # Connectivity reordering for gmsh to VTK: the two share vertex and
        # most mid-edge nodes, but gmsh's last two are swapped relative to
        # VTK_QUADRATIC_TETRA.
        cells_numpy = result.elements.astype(np.int64)[:, VTK_TET10_FROM_GMSH]

        connectivity = np.hstack([np.full((num_elems, 1), cells_numpy.shape[1], dtype=np.int64), cells_numpy]).flatten()

        cell_array = vtk.vtkCellArray()
        cell_array.SetCells(num_elems, numpy_support.numpy_to_vtkIdTypeArray(connectivity, deep=1))

        cell_types = np.full(num_elems, vtk.VTK_QUADRATIC_TETRA, dtype=np.uint8)

        vtk_grid = vtk.vtkUnstructuredGrid()
        vtk_grid.SetPoints(vtk_points)
        vtk_grid.SetCells(numpy_support.numpy_to_vtk(cell_types, deep=1, array_type=vtk.VTK_UNSIGNED_CHAR), cell_array)

        point_data = vtk_grid.GetPointData()
        point_data.AddArray(self._to_vtk_array(result.displacements.reshape(-1, 3), "Displacement"))
        # Lets ParaView's Warp By Vector find the field without being told.
        point_data.SetActiveVectors("Displacement")

        for i, label in enumerate(TENSOR_LABELS):
            if result.strains is not None:
                point_data.AddArray(self._to_vtk_array(result.strains[:, i], f"Strain_{label}"))
            if result.stresses is not None:
                point_data.AddArray(self._to_vtk_array(result.stresses[:, i], f"Stress_{label}"))

        return vtk_grid

    def export(self, result: FEMResult, filename: Path) -> None:
        """
        Export the data to a vtu file.

        :param result: FEM results.
        :type result: FEMResult
        :param filename: Filename to save to.
        :type filename: Path
        """
        logger.info(f"Building VTK grid: {result.nodes.shape[0]} nodes, {result.elements.shape[0]} elements")

        try:
            vtk_grid = self._build_grid(result)
        except ValueError as e:
            logger.error(f"Data shape mismatch during VTK export for {filename}: {e}")
            raise
        except (TypeError, AttributeError) as e:
            logger.error(f"Data type error (NoneType or invalid dtype) in VTK conversion: {e}")
            raise

        filename.parent.mkdir(parents=True, exist_ok=True)

        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(str(filename))
        writer.SetInputData(vtk_grid)

        if writer.Write() != 1:
            raise OSError(f"VTK writer failed to write file: {filename}")

        logger.success(f"Results written to {filename}")
