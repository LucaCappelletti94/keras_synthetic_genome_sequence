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
        # Get the boolean masks for the original positions that contain the gaps
        # for the given index
        masks = self._original_indices == indices[:, None]
        # Get the mask to drop the rows where none of the indices
        # considered for this specific batch is present
        considered_rows = masks.any(axis=0)
        # Drop rows where none of the indices is present
        indices_masks = masks[:, considered_rows]
        coordinates = self._coordinates[considered_rows]
        # Making a deep copy of y, since we are going to edit the copy.
        x = np.copy(y)
        # For i-th row of current batch we apply the nucletides mask
        for i, indices_mask in enumerate(indices_masks):
            x[i, coordinates[indices_mask]] = 0.25
        return x, y