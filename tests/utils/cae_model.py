from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Reshape, Conv2DTranspose


def cae_model():
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