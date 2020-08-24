import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, activation='elu', input_shape=(32, ), # We can specify initializers
          kernel_initializer='he', bias_initializer='zeros'),
    Dense(1, activation='sigmoid')
])

# Model compiler
# it defines loss function and optimizer
# also contains metrics of performance
model.compile(
    optimizer='sgd',
    loss='binary_crossentropy',
    metrics=['accuracy'] # Pass list of metrics
)

# We can also call object or functions directly
# to change some hyper-parameters
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=3e-3, momentum=0.9),
    loss=tf.keras.losses.binary_crossentropy(from_logits=False),
    metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.7)]
)