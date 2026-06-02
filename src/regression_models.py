import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def simple_regressr(feature_size) -> keras.Sequential:
    """
    returns a simple regression model
    with a single embedding layer and
    a single prediction layer
    """
    model = keras.Sequential([
        layers.Dense(feature_size, input_shape=(feature_size,)),
        # softplus activation to keep positive
        layers.Dense(feature_size, activation='softplus'),
        # regreesion layear, no activation
        layers.Dense(1)
    ])
    return model

def embedded_regressor(feature_size) -> keras.Sequential:
    """
    regression model with extra hidden spaces
    """
    model = keras.Sequential([
        layers.Dense(feature_size, input_shape=(feature_size,)),
        layers.Dense(int(feature_size/2), activation='softplus'),
        layers.Dense(int(feature_size/2), activation='softplus'),
        layers.Dense(1)
    ])
    return model

def more_embedded_regressor(feature_size) -> keras.Sequential:
    """
    regression model with extra hidden spaces
    """
    model = keras.Sequential([
        layers.Dense(feature_size, input_shape=(feature_size,)),
        layers.Dense(feature_size, activation='softplus'),
        layers.Dense(int(feature_size/2), activation='softplus'),
        layers.Dense(1)
    ])
    return model

def hi_embedded_regressor(feature_size) -> keras.Sequential:
    """
    even more embedding space
    """
    model = keras.Sequential([
        layers.Dense(feature_size, input_shape=(feature_size,)),
        # compress and expand hidden space
        # relue activation on hidden space for more discrete activation
        layers.Dense(feature_size*2, activation='softplus'),
        layers.Dense(feature_size, activation='softplus'),
        layers.Dense(int(feature_size/2), activation='softplus'),
        # softplus activation to keep positive
        layers.Dense(feature_size, activation='softplus'),
        # regreesion layear, no activation
        layers.Dense(1)
    ])
    return model

def compile_train_checkpoint(model,
                             X_train,
                             X_test,
                             y_train,
                             y_test,
                             epochs: int=2000,
                             filepath='',
                             learning_rate=0.005,
                             loss='mse'):
    """
    compiles model and trains it,
    returns the training history
    """
    callback = keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        verbose=0,
        validation_data=(X_test, y_test),
        callbacks=[callback]
    )
    return history

