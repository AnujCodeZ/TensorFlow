import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([Dense(1, activation='sigmoid', input_shape=(12,))])
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train)

# It runs model on testset
# and returns loss and accuracy
# if mae metrics it also returns that
loss, accuracy = model.evaluate(X_test, y_test)

# Prediction
# X_sample: (num_samples, 12)
# must pass numpy array or list
# for only one sample X_sample: (1, 12)
pred = model.predict(X_sample)