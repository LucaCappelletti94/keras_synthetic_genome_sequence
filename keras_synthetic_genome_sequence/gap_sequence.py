from typing import Union, Dict, Tuple
import pandas as pd
import numpy as np
from keras_bed_sequence import BedSequence
from keras_mixed_sequence.utils import NumpySequence
from .utils import generate_synthetic_gaps


class GapSequence(BedSequence):
    def __init__(
        self,
        assembly: str,
        bed: Union[pd.DataFrame, str],
        gaps_mean: np.ndarray,
        gaps_covariance: np.ndarray,
        gaps_threshold: float = 0.5,
        batch_size: int = 32,
        verbose: bool = True,
        seed: int = 42,
        elapsed_epochs: int = 0,
        genome_kwargs: Dict = None
    ):
        """Return new GapSequence object.

        Parameters
        ----------------------------
        assembly: str,
            Genomic assembly from ucsc from which to extract sequences.
            For instance, "hg19", "hg38" or "mm10".
        bed: Union[pd.DataFrame, str],
            Either path to file or Pandas DataFrame containing minimal bed columns,
            like "chrom", "chromStart" and "chromEnd".
        gaps_mean: np.ndarray,
            Mean of the multivariate Gaussian distribution to use for generating
            the gaps in the sequences. Length of the sequences must match with
            length of the mean vector.
        gaps_covariance: np.ndarray,
            Covariance matrix of the multivariate Gaussian distribution to use
            for generating the gaps in the sequences.
            Length of the sequences must match with length of the mean vector.
        gaps_threshold: float,
            Threshold for casting the multivariate Gaussian distribution to
            a binomial multivariate distribution.
        batch_size: int = 32,
            Batch size to be returned for each request.
            By default is 32.
        verbose: bool = True,
            Whetever to show a loading bar.
        seed: int = 42,
            Starting seed to use if shuffling the dataset.
        elapsed_epochs: int = 0,
            Number of elapsed epochs to init state of generator.
        genome_kwargs: Dict = None,
            Parameters to pass to the Genome object.

        Returns
        --------------------
        Return new GapSequence object.
        """
        super().__init__(
            assembly=assembly,
            bed=bed,
            batch_size=batch_size,
            verbose=verbose,
            seed=seed,
            elapsed_epochs=elapsed_epochs,
            genome_kwargs=genome_kwargs,
        )
        if len(gaps_mean) != self.window_length:
            raise ValueError(
                "Given gaps mean vector length ({mean_len}) does not batch given bed file window length ({window_len}).".format(
                    mean_len=len(gaps_mean),
                    window_len=self.window_length,
                )
            )
        if len(gaps_covariance) != self.window_length:
            raise ValueError(
                "Given gaps covariance vector length ({covariance_len}) does not batch given bed file window length ({window_len}).".format(
                    covariance_len=len(gaps_covariance),
                    window_len=self.window_length,
                )
            )
        # Rendering the gaps coordinates
        self._gaps_coordinates = generate_synthetic_gaps(
            gaps_mean,
            gaps_covariance,
            self.samples_nuber,
            chunk_size=50000,
            threshold=gaps_threshold,
            seed=seed
        )
        # Rendering the starting gaps index, which
        # will be shuffled alongside the bed file.
        self._gaps_index = NumpySequence(
            np.arange(self.samples_nuber),
            batch_size=batch_size,
            seed=seed,
            elapsed_epochs=elapsed_epochs
        )

    def on_epoch_end(self):
        """Shuffle private bed object on every epoch end."""
        super().on_epoch_end()
        self._gaps_index.on_epoch_end()

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
        masks = self._gaps_coordinates[
            np.in1d(self._gaps_coordinates[:, 0], indices)
        ]
        x = np.copy(y)
        for i, index in enumerate(indices):
            gap_indices = masks[masks[:, 0] == index][:, 1]
            x[i, gap_indices, :] = 0.25
        return x, y