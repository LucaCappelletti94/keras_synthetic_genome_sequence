import os
from .utils import cae_model, get_test_bed
from keras_synthetic_genome_sequence import MultivariateGapWindowsSequence
from keras_synthetic_genome_sequence.utils import get_gaps_statistics
from ucsc_genomes_downloader import Genome
import numpy as np
import pandas as pd


def test_multivariate_gap_windows_sequence():
    hg19 = Genome("hg19", chromosomes=["chr1", "chr2", "chr3"])

    _, mean, covariance = get_gaps_statistics(
        hg19,
        100,
        200
    )

    gap_sequence = MultivariateGapWindowsSequence(
        assembly=hg19,
        bed=get_test_bed(),
        gaps_mean=mean,
        gaps_covariance=covariance,
        batch_size=32
    )

    x1, y1 = gap_sequence[0]
    x2, y2 = gap_sequence[0]

    assert (x1 == 0.25).any()
    assert set((0.25, 0.0, 1.0)) == set(np.unique(x1))
    assert (x1 == x2).all()
    assert (y1 == y2).all()

    assert x1.shape == y1.shape
    assert x1.shape == (gap_sequence.batch_size, 200, 4)

    cae_model().fit_generator(
        gap_sequence,
        steps_per_epoch=gap_sequence.steps_per_epoch,
        epochs=2,
        verbose=0,
        shuffle=True
    )
