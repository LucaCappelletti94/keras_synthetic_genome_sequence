import os
import pytest
import numpy as np
from keras_synthetic_genome_sequence import GapSequence


def test_misuse():
    with pytest.raises(ValueError):
        GapSequence(
            "hg19",
            "{cwd}/test.bed".format(
                cwd=os.path.dirname(os.path.abspath(__file__))
            ),
            gaps_mean=np.random.randint(2, size=190),
            gaps_covariance=np.random.randint(2, size=(200, 200)),
            batch_size=32
        )
    with pytest.raises(ValueError):
        GapSequence(
            "hg19",
            "{cwd}/test.bed".format(
                cwd=os.path.dirname(os.path.abspath(__file__))
            ),
            gaps_mean=np.random.randint(2, size=200),
            gaps_covariance=np.random.randint(2, size=(190, 200)),
            batch_size=32
        )
