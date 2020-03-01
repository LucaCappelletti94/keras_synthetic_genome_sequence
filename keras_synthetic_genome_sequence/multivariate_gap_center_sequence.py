"""Keras Sequence that returns tuples of nucleotide sequences, one with multivariate synthetic gaps and the other without as ground truth."""
from typing import Union, Dict, Tuple
import pandas as pd
import numpy as np
from .multivariate_gap_windows_sequence import MultivariateGapWindowsSequence
from .utils import generate_synthetic_gaps


class MultivariateGapCenterSequence(MultivariateGapWindowsSequence):
    """
    Keras Sequence that returns tuples of nucleotide sequences,
    one with multivariate synthetic gaps and the other with the
    values of the chromosome in the middle.
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
        x, y = super().__getitem__(idx)
        return x, y[:, self.window_length//2]
