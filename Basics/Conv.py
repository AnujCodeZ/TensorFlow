from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D

model = Sequential([
    Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPool2D(pool_size=(3, 3)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# For same padding
model = Sequential([
    Conv2D(filters=16, kernel_size=(3, 3), padding='SAME',
           activation='relu', input_shape=(32, 32, 3)),
    MaxPool2D(pool_size=(3, 3)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Kernel size or pool size as single integer
model = Sequential([
    Conv2D(filters=16, kernel_size=3, activation='relu', input_shape=(32, 32, 3)),
    MaxPool2D(pool_size=3),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Strides (steps of convolution)
model = Sequential([
    Conv2D(filters=16, kernel_size=(3, 3), strides=2, activation='relu', input_shape=(32, 32, 3)),
    MaxPool2D(pool_size=(3, 3)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Data format
# position of channel in image size
# By default channels last
model = Sequential([
    Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(3, 32, 32), data_format='channels_first'),
    MaxPool2D(pool_size=(3, 3), data_format='channels_first'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.summary()