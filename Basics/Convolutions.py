import tensorflow as tf
from tensorflow import keras

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = \
    mnist.load_data()

train_images = train_images.reshape(-1, 28, 28, 1)
train_images = train_images / 255.0
test_images = test_images.reshape(-1, 28, 28, 1)
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy')

model.fit(train_images, train_labels, epochs=100)

model.evaluate(test_images, test_labels)