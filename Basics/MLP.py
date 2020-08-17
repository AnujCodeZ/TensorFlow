from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Softmax

# Creating an instance of Sequential class
# We pass a list of layers into this
model = Sequential([
    Dense(units=64, activation='relu', input_shape=(784, )),
    Dense(units=10, activation='softmax')
])

# Alternative way
# Used in applying conditions on layers
model = Sequential()

model.add(Dense(units=64, activation='relu', input_shape=(784, )))
model.add(Dense(units=10, activation='softmax'))

# Alternative input layer
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(units=64, activation='relu'),
    Dense(units=10, activation='softmax')
])

# Alternative softmax activation
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(units=64, activation='relu'),
    Dense(units=10),
    Softmax()
])

# Can define names to layers also
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(units=64, activation='relu', name='layer_1'),
    Dense(units=10, activation='softmax')
])

# Details of a model.
print(model.summary())