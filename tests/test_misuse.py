import os
import pytest
import numpy as np
from .utils import get_test_bed
from keras_synthetic_genome_sequence import MultivariateGapCenterSequence


def test_misuse():
    with pytest.raises(ValueError):
        MultivariateGapCenterSequence(
            "hg19",
            get_test_bed(),
            gaps_mean=np.random.randint(2, size=190),
            gaps_covariance=np.random.randint(2, size=(200, 200)),
            batch_size=32
        )
    with pytest.raises(ValueError):
        MultivariateGapCenterSequence(
            "hg19",
            get_test_bed(),
            gaps_mean=np.random.randint(2, size=200),
            gaps_covariance=np.random.randint(2, size=(190, 200)),
            batch_size=32
        )
