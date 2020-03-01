"""Keras Sequence that returns tuples of nucleotide sequences, one with multivariate synthetic gaps and the other without as ground truth."""
from typing import Union, Dict, Tuple
import pandas as pd
import numpy as np
from .multivariate_gap_sequence import MultivariateGapSequence
from .utils import generate_synthetic_gaps
from numba import njit


@njit
def add_gaps(gaps_coordinates: dict, indices: np.ndarray, y: np.ndarray):
    # Making a deep copy of y, since we are going to edit the copy.
    x = np.copy(y)
    for i in range(indices.shape[0]):
        x[i][gaps_coordinates[indices[i]]] = 0.25
    return x


class MultivariateGapWindowsSequence(MultivariateGapSequence):
    """
    Keras Sequence that returns tuples of nucleotide sequences,
    one with multivariate synthetic gaps and the other without as ground truth.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Compiling NUMBA function
        add_gaps(self._gaps_coordinates,
                 self._gaps_index[0], super().__getitem__(0))

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return batch corresponding to given index.

        Parameters
        ---------------
        idx: int,
            Index corresponding to batch to be rendered.

        Returns
        ---------------
        Return Tuple containing X and Y numpy arrays corresponding to given batch index.
        """
        # Retrieves the sequence from the bed generator
        y = super().__getitem__(idx)
        # For i-th row of current batch we apply the nucletides mask
        x = add_gaps(self._gaps_coordinates, self._gaps_index[idx], y)
        return x, y
