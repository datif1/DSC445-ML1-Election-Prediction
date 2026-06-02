Training with: 

raw_data = '../data/processed/2020_joined_norm_jeong.csv'

```python
regression_model = tf.keras.Sequential([
    layers.Dense(feature_size, input_shape=(feature_size,)),
    # softplus activation to keep positive
    layers.Dense(feature_size, activation='softplus'),
    # regreesion layear, no activation
    layers.Dense(1)
])

filepath='../models/best_nn_linear_regression.h5',
```

R2 Score:  tf.Tensor(0.7284051, shape=(), dtype=float32)
MSE Score:  tf.Tensor(0.006850635, shape=(), dtype=float32)

```python
# two hidden layer with compressing embedding space by half
comp_reg_model = tf.keras.Sequential([
    layers.Dense(feature_size, input_shape=(feature_size,)),
    # softplus activation to keep positive
    layers.Dense(int(feature_size/2), activation='softplus'),
    layers.Dense(int(feature_size/2), activation='softplus'),
    # regreesion layear, no activation
    layers.Dense(1)
])

filepath='../models/best_nn_comp_linear_regression.h5',

```

R2 Score:  tf.Tensor(0.7409699, shape=(), dtype=float32)
MSE Score:  tf.Tensor(0.006533704, shape=(), dtype=float32)

```python
# more layers!
more_reg_model = tf.keras.Sequential([
    layers.Dense(feature_size, input_shape=(feature_size,)),
    # softplus activation to keep positive
    # compress and expand hidden space
    layers.Dense(feature_size, activation='softplus'),
    layers.Dense(int(feature_size/2), activation='softplus'),
    layers.Dense(feature_size, activation='softplus'),
    # regreesion layear, no activation
    layers.Dense(1)
])

 filepath='../models/best_nn_more_linear_regression.h5',
```

R2 Score:  tf.Tensor(0.7544497, shape=(), dtype=float32)
MSE Score:  tf.Tensor(0.006193694, shape=(), dtype=float32)

```python
# high count of hidden state variables
hi_reg_model = tf.keras.Sequential([
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

filepath='../models/best_nn_hi_linear_regression.h5',
```

```python
# callback for saving weights with validation
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='../models/best_nn_more_linear_regression.h5',
    monitor='val_accuracy', # use validation as monitoring accuracy
    save_best_only=True,
    mode='max'              
)
```

```python

hi_reg_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
    loss='mse'
)

hi_training_history = hi_reg_model.fit(
    X_train,
    y_train,
    epochs = 2000,
    verbose = 0,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint_callback]
)

```

R2 Score:  tf.Tensor(0.75437826, shape=(), dtype=float32)
MSE Score:  tf.Tensor(0.0061954954, shape=(), dtype=float32)