from matplotlib import image
import numpy as np
from os import listdir
import csv

loaded_images = list()
y = list()
load_tags = list()
with open('../Datasets/images/metadata.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            load_tags.append(row[1])
            line_count += 1
counter = 0
for folder in listdir('../Datasets/images/'):
    if folder == '.DS_Store' or folder == 'metadata.csv':
        continue
    for filename in listdir('../Datasets/images/' + folder):
    	# load image
    	img_data = image.imread('../Datasets/images/'+ folder + '/' + filename)
    	# store loaded image
    	loaded_images.append(img_data)
    print("Currently processing: " + folder)
    y+= [load_tags[counter]] * (len(loaded_images) - len(y))
    counter += 1
Y_train = np.asarray(y)
loaded_images= np.asarray(loaded_images)
X_train_flatten = loaded_images.reshape(loaded_images.shape[0], -1).T
X_train = X_train_flatten/255.
#Y_train = Y_train.reshape(Y_train.shape[0], 1)
# use this reshape when using the tensorflow model
Y_train = Y_train.reshape(1, Y_train.shape[0])
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
np.save('x_train', X_train)
np.save('y_train', Y_train)
