import tensorflow as tf
from tensorflow import keras

class MyCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss')<0.3):
            print('Reached 70% accuracy so stop training.')
            self.model.stop_training = True

callbacks = MyCallback()

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = \
    mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy')

model.fit(train_images, train_labels, epochs=100, callbacks=[callbacks])

model.evaluate(test_images, test_labels)