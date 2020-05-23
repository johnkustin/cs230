import numpy as np


X_train = np.load('y_train.npy')
print(list(X_train).count('REAL'))
print(X_train.shape)
