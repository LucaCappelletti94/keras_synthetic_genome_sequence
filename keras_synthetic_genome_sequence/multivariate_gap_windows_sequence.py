"""Keras Sequence that returns tuples of nucleotide sequences, one with multivariate synthetic gaps and the other without as ground truth."""
from typing import Union, Dict, Tuple
import pandas as pd
import numpy as np
from .multivariate_gap_sequence import MultivariateGapSequence
from .utils import generate_synthetic_gaps


class MultivariateGapWindowsSequence(MultivariateGapSequence):
    """
    Keras Sequence that returns tuples of nucleotide sequences,
    one with multivariate synthetic gaps and the other without as ground truth.
    """

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
        # Retrieve the indices corresponding to the gaps for the current batchsize
        indices = self._gaps_index[idx]
        # Extract the gaps curresponding to given indices
        masks = self._gaps_coordinates[:, 0] == indices[:, None]
        considered_rows = masks.any(axis=0)
        indices_masks = masks[:, considered_rows]
        positions = self._gaps_coordinates[:, 1][considered_rows]
        # Making a deep copy of y, since we are going to edit the copy.
        x = np.copy(y)
        # For every j-th index curresponding to the i-th row of current batch
        for i, indices_mask in enumerate(indices_masks):
            x[i, positions[indices_mask]] = 0.25
        return x, y