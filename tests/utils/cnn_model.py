from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Reshape, Dense, Flatten

def cnn_model():
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
