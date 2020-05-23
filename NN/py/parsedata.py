from matplotlib import image
import numpy as np
from os import listdir
import csv

loaded_images = list()
y = list()
load_tags = {}
with open('../Datasets/images/metadata.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            load_tags.update({row[0]: row[1]})
            line_count += 1
counter = 0
for folder in listdir('../Datasets/images/'):
    if counter > 15:
        break
    if folder == '.DS_Store' or folder == 'metadata.csv':
        continue
#    for filename in listdir('../datasets/faces-sample/' + folder):
    filename = listdir('../datasets/faces-sample/' + folder)[0]
    # load image
    img_data = image.imread('../datasets/faces-sample/'+ folder + '/' + filename)
    # store loaded image
    loaded_images.append(img_data)


    print("Currently processing: " + folder)
    if (load_tags[folder + '.mp4'] == 'REAL'):
        l = [[1, 0]] * (len(loaded_images) - len(y))
        y.extend(l)
    else:
        l = [[0, 1]] * (len(loaded_images) - len(y))
        y.extend(l)
    # y+= [load_tags[folder + '.mp4']] * (len(loaded_images) - len(y))
    counter += 1
Y_train = np.asarray(y)
loaded_images= np.asarray(loaded_images)
print(loaded_images.shape)
X_train = loaded_images/255.
Y_train = Y_train
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
np.save('x_train_small', X_train)
np.save('y_train_small', Y_train)
