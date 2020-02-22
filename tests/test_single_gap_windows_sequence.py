import os
from .utils import cae_model, get_test_bed
from keras_synthetic_genome_sequence import SingleGapWindowsSequence
from keras_synthetic_genome_sequence.utils import get_gaps_statistics
import numpy as np
import pandas as pd


def test_single_gap_windows_sequence():
    gap_sequence = SingleGapWindowsSequence(
        assembly="hg19",
        bed=get_test_bed(),
        batch_size=32
    )

    x1, y1 = gap_sequence[0]
    x2, y2 = gap_sequence[0]

    assert np.isclose(x1, 0.25).sum() == 4*gap_sequence.batch_size
    assert (np.isclose(x1, 0.25).all(axis=2).argmax(axis=1) == 100).all()
    assert 0.25 not in y1
    assert (x1 == x2).all()
    assert x1.shape == y1.shape
    assert (y1 == y2).all()

    cae_model().fit_generator(
        gap_sequence,
        steps_per_epoch=gap_sequence.steps_per_epoch,
        epochs=2,
        verbose=0,
        shuffle=True
    )
