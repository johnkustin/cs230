import math
import numpy as np
#import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import *
from tf_helper_funcs import *

np.random.seed(1)

#X_full = np.load("x_train.npy")
#Y_full = np.load("y_train.npy")
#Y_full = (Y_full == 'REAL')
X_full = np.load('/data/testdata/x_dataset_half.npz')
Y_full = np.load('/data/testdata/y_dataset_half.npz')
X_full_dat = X_full['arr_0']
Y_full_dat = Y_full['arr_0']
X_full.close()
Y_full.close()
X_train_dat = X_full_dat[251:2500,:,:,:]
Y_train_dat = Y_full_dat[251:2500,:]
Y_test_dat = Y_full_dat[:250,:]
X_test_dat = X_full_dat[:250,:,:,:]

X_train_dat = X_train_dat.reshape(X_train_dat.shape[0], -1)
X_test_dat = X_test_dat.reshape(X_test_dat.shape[0],-1)

Y_train_dat = Y_train_dat[:,0]
Y_test_dat = Y_test_dat[:,0]

Y_test_dat = Y_test_dat.reshape(len(Y_test_dat),1)
Y_train_dat = Y_train_dat.reshape(len(Y_train_dat),1)

print('X test shape: ')
print(X_test_dat.shape)
print('X train shape: ' )
print(X_train_dat.shape)
print('Y test shape: ' )
print(Y_test_dat.shape)
print('Y train shape: ' )
print(Y_train_dat.shape)
def model(X_train = X_train_dat, Y_train = Y_train_dat, X_test = X_test_dat, Y_test = Y_test_dat, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    8201 total examples = 7701 train + 500 test
    
    Arguments:
    X_train -- training set, of shape (input size = 76800, number of training examples = 7701)
    Y_train -- test set, of shape (output size = 1, number of training examples = 7701)
    X_test -- training set, of shape (input size = 76800, number of training examples = 500)
    Y_test -- test set, of shape (output size = 1, number of test examples = 500)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (m, n_x) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)

    # Initialize parameters
    parameters = initialize_parameters()
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, parameters)
    
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z3, Y)
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                           # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train.T, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                
                epoch_cost += minibatch_cost / minibatch_size

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per fives)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        plt.savefig('DNN.png')

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        return parameters
parameters = model(X_train = X_train_dat,Y_train = Y_train_dat.T, X_test = X_test_dat, Y_test = Y_test_dat, minibatch_size=64)
