
#import time
import numpy as np
#import h5py
import matplotlib.pyplot as plt
#import scipy
#from PIL import Image
#from scipy import ndimage
from dnn_app_utils_v3 import *

# get_ipython().magic('matplotlib inline')
# plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'
# 
# get_ipython().magic('load_ext autoreload')
# get_ipython().magic('autoreload 2')

train_x, train_y, test_x, test_y, classes = load_data()
train_x = np.asarray(train_x,dtype=np.float64)
train_y = np.asarray(train_y)
print(train_y.shape)
train_y = (train_y == 'REAL')
print(train_y)


print ("train_x's shape: " + str(train_x.shape))

num_px = train_x.shape[0] #include *3 for RGB

### CONSTANTS ###
layers_dims = [num_px, 20, 7, 5, 1] #  4-layer model
# GRADED FUNCTION: L_layer_model

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=True):#lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    costs = []                         # keep track of cost
    
    # Parameters initialization. (≈ 1 line of code)
    ### START CODE HERE ###
    parameters = initialize_parameters_deep(layers_dims)
    ### END CODE HERE ###
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):
        
        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        ### START CODE HERE ### (≈ 1 line of code)
        AL, caches = L_model_forward(X, parameters)
        ### END CODE HERE ###
        
        # Compute cost.
        ### START CODE HERE ### (≈ 1 line of code)
        cost = compute_cost(AL,Y)
        ### END CODE HERE ###
         
        # Backward propagation.
        ### START CODE HERE ### (≈ 1 line of code)
        grads = L_model_backward(AL,Y,caches)
        ### END CODE HERE ###
        
        # Update parameters.
        ### START CODE HERE ### (≈ 1 line of code)
        parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###
                
        # Print the cost every 100 training example
        if print_cost:
            print ("Cost after iteration %i: %f" %(i, cost))
        costs.append(cost)
        print("Finished iteration ", i) 
    # plot the cost
   # plt.plot(np.squeeze(costs))
   # plt.ylabel('cost')
   # plt.xlabel('iterations (per hundreds)')
   # plt.title("Learning rate =" + str(learning_rate))
   # plt.show()
    
    np.save('parameters',parameters)
    np.save('costs',costs)
    return parameters


# You will now train the model as a 4-layer neural network. 
# 
# Run the cell below to train your model. The cost should decrease on every iteration. It may take up to 5 minutes to run 2500 iterations. Check if the "Cost after iteration 0" matches the expected output below, if not click on the square (⬛) on the upper bar of the notebook to stop the cell and try to find your error.

# In[17]:

parameters = L_layer_model(train_x, train_y.T, layers_dims, num_iterations = 2500, print_cost = True)

pred_train = predict(train_x, train_y, parameters)
# pred_test = predict(test_x, test_y, parameters)

# print_mislabeled_images(classes, test_x, test_y, pred_test)

## START CODE HERE ##
# my_image = "my_image.jpg" # change this to the name of your image file 
# my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)
# ## END CODE HERE ##
# 
# fname = "images/" + my_image
# image = np.array(ndimage.imread(fname, flatten=False))
# my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
# my_image = my_image/255.
# my_predicted_image = predict(my_image, my_label_y, parameters)
# 
# plt.imshow(image)
# print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
# 

# **References**:
# 
# - for auto-reloading external module: http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
