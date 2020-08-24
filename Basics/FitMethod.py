import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, activation='elu', input_shape=(32, )),
    Dense(100, activation='softmax')
])

model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# We can start training by fit method
# by passing examples in it
# X_train: (num_samples, num_features)
# y_train: (num_samples, num_classes) they are one hot
# If y_train: (num_samples, ) then loss is
# sparse_categorical_crossentropy
# history is callback object
history = model.fit(X_train, y_train, epochs=10, batch_size=16)