"""Typed access to HDF5 nodes."""

import h5py


def dataset(node: h5py.File | h5py.Group, key: str) -> h5py.Dataset:
    """Return the dataset at 'key', failing clearly if it is missing or not a dataset."""
    item = node[key]
    if not isinstance(item, h5py.Dataset):
        raise TypeError(f"{key!r} is a {type(item).__name__}, expected a dataset")
    return item


def group(node: h5py.File | h5py.Group, key: str) -> h5py.Group:
    """Return the group at 'key', failing clearly if it is missing or not a group."""
    item = node[key]
    if not isinstance(item, h5py.Group):
        raise TypeError(f"{key!r} is a {type(item).__name__}, expected a group")
    return item
