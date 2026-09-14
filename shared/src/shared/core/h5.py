"""Typed access to HDF5 datasets, groups and attributes."""

from typing import Any, cast

import h5py
from numpy.typing import NDArray


def dataset(node: h5py.File | h5py.Group, key: str) -> h5py.Dataset:
    """Return the dataset at 'key', failing clearly if it is not a dataset."""
    item = node[key]
    if not isinstance(item, h5py.Dataset):
        raise TypeError(f"{key!r} is a {type(item).__name__}, expected a dataset")
    return item


def group(node: h5py.File | h5py.Group, key: str) -> h5py.Group:
    """Return the group at 'key', failing clearly if it is not a group."""
    item = node[key]
    if not isinstance(item, h5py.Group):
        raise TypeError(f"{key!r} is a {type(item).__name__}, expected a group")
    return item


def attr(node: h5py.File | h5py.Group, key: str) -> float | int | str | NDArray[Any]:
    """Return the attribute at 'key', failing clearly if it is missing."""
    if key not in node.attrs:
        raise KeyError(f"{node.file.filename} has no attribute {key!r}")
    return cast("float | int | str | NDArray[Any]", node.attrs[key])
