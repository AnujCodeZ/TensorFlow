import tensorflow as tf
import numpy as np

class NeuralNetwork:

    # Constructor
    def __init__(self, layers):

        # Layers
        self.Layers = layers
        self.L = len(layers)
        self.num_features = layers[0]
        self.num_labels = layers[-1]
        
        # Parameters
        self.weights = {}
        self.bias = {}
        # Parameters derivatives
        self.dW = {}
        self.db = {}

        self.initialize_params()
    
    # Initializing parameters
    def initialize_params(self):
        
        for i in range(1, self.L):

            self.weights = tf.Variable(tf.random.normal(shape=(self.Layers[i], self.Layers[i-1])))
            self.bias = tf.Variable(tf.zeros(shape=(self.Layers[i], 1)))
    
    # Forward pass
    def forward(self, X):

        A = tf.convert_to_tensor(X, dtype=tf.float32)
        for i in range(1, self.L):

            Z = tf.add(tf.matmul(A, tf.transpose(self.weights[i])), tf.transpose(self.bias[i]))
            if i != self.L-1: # Not last layer

                A = tf.nn.relu(Z)
            
            else:

                A = Z
        
        return A
    
    # Cost function
    def compute_cost(self, A, Y):

        loss = tf.nn.softmax_cross_entropy_with_logits(Y, A)

        return tf.reduce_mean(loss)
    

    # Backward pass
    def backward(self, X, Y):

        with tf.GradientTape(persistent=True) as tape:

            A = self.forward(X)
            loss = self.compute_cost(A, Y)
        
        for i in range(1, self.L):

            self.dW[i] = tape.gradient(loss, self.weights[i])
            self.db[i] = tape.gradient(loss, self.bias[i])
        
        del tape

        return loss

    # Update parameters
    def update_params(self, lr):

        for i in range(1, self.L):

            self.weights[i].assign_sub(lr * self.dW[i])
            self.bias[i].assign_sub(lr * self.db[i])
    
    # Prediction
    def predict(self, X):

        A = self.forward(X)

        return tf.argmax(tf.nn.softmax(A), axis=1)
    
    # Summary function
    def summary(self):

        num_params = 0
        for i in range(1, self.L):

            num_params += self.weights[i].shape[0] * self.weights[i].shape[1]
            num_params += self.bias[i].shape[0]

        print("Input Features:", self.num_features)
        print("Number of classes:", self.num_labels)
        print("Hidden Layers:")
        print("----------")
        for i in range(1, self.L-1):
            print("Layer {}, Units {}".format(i, self.Layers[i]))
        print("----------")
        print("Total number of parameters:", num_params)
    
    # Training on batch
    def train_on_batch(self, X, Y, lr):

        X = tf.convert_to_tensor(X, tf.float32)
        Y = tf.convert_to_tensor(Y, tf.float32)

        loss = self.backward(X, Y)
        self.update_params(lr)

        return loss.numpy()
    
    # Training
    def train(self, x_train, y_train, x_test, y_test, epochs, batch_size, lr):

        steps_per_epoch = int(x_train[0]/batch_size)
        train_loss = []
        val_acc = []

        for e in range(epochs):
            epoch_loss = 0
            print("Epoch: {}".format(e), end='.')
            for i in range(steps_per_epoch):

                x_batch = x_train[i*batch_size:(i+1)*batch_size]
                y_batch = y_train[i*batch_size:(i+1)*batch_size]

                batch_loss = self.train_on_batch(x_batch, y_batch. lr)
                epoch_loss += batch_loss

                if i%int(steps_per_epoch/10) == 0:
                    print(end='.')
            
            train_loss.append(epoch_loss/steps_per_epoch)

            val_preds = self.predict(x_test)
            val_acc.append(np.mean(np.argmax(y_test, axis=1) == val_preds.numpy()))
            print("Validation Accuracy:", val_acc[-1])

        return train_loss, val_acc

