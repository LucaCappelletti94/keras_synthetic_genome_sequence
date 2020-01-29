import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Reshape, Conv2DTranspose
from keras_synthetic_genome_sequence import GapSequence
from keras_synthetic_genome_sequence.utils import get_gaps_statistics


def build_model():
    model = Sequential([
        Reshape((200, 4, 1)),
        Conv2D(16, kernel_size=3, activation="relu"),
        Conv2DTranspose(1, kernel_size=3, activation="relu"),
        Reshape((-1, 200, 4))
    ])
    model.compile(
        optimizer="nadam",
        loss="MSE"
    )
    return model


def test_model_denoiser():
    batch_size = 32
    _, mean, covariance = get_gaps_statistics(
        "hg19",
        100,
        200
    )
    gap_sequence = GapSequence(
        "hg19",
        "{cwd}/test.bed".format(
            cwd=os.path.dirname(os.path.abspath(__file__))
        ),
        gaps_mean=mean,
        gaps_covariance=covariance,
        batch_size=batch_size
    )
    x1, y1 = gap_sequence[0]
    x2, y2 = gap_sequence[0]

    assert (x1 == 0.25).any()
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