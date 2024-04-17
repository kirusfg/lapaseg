import keras


def mlp(device: str = "cuda"):
    model = keras.Sequential(name="mlp")

    model.add(keras.layers.Input((1920,)))
    model.add(keras.layers.Dense(2048))
    model.add(keras.layers.Dense(1536))
    model.add(keras.layers.Dense(1024))
    model.add(keras.layers.Dense(512))

    model.add(keras.layers.Dense(11))
    model.add(keras.layers.Softmax())

    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    model.to(device)

    return model
