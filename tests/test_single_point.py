import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Reshape, Conv2DTranspose, Flatten, Dense
from keras_synthetic_genome_sequence import SingleGapSequence
from keras_synthetic_genome_sequence.utils import get_gaps_statistics
import numpy as np
import pandas as pd


def build_model():
    model = Sequential([
        Reshape((200, 4, 1)),
        Conv2D(1, kernel_size=3, activation="relu"),
        Flatten(),
        Dense(4, activation="sigmoid")
    ])
    model.compile(
        optimizer="nadam",
        loss="categorical_crossentropy"
    )
    return model


def test_model_denoiser():
    batch_size = 32
    bed = pd.read_csv("{cwd}/test.bed".format(
        cwd=os.path.dirname(os.path.abspath(__file__))
    ), sep="\t")
    gap_sequence = SingleGapSequence(
        "hg19",
        bed,
        batch_size=batch_size
    )
    x1, y1 = gap_sequence[0]
    x2, y2 = gap_sequence[0]

    assert np.isclose(x1, 0.25).sum() == 4*batch_size
    assert (np.isclose(x1, 0.25).all(axis=2).argmax(axis=1) == 100).all()
    assert 0.25 not in y1
    assert (x1 == x2).all()
    assert (y1 == y2).all()

    model = build_model()
    model.fit_generator(
        gap_sequence,
        steps_per_epoch=gap_sequence.steps_per_epoch,
        epochs=2,
        verbose=0,
        shuffle=True
    )
