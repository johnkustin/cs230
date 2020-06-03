from matplotlib import image
import numpy as np
from os import listdir
import csv
from os import chdir

loaded_images = list()
y = list()
load_tags = {}
path = '/home/ubuntu/cs230/140k-real-and-fake-faces/'
#with open('../Datasets/images/metadata.csv') as csv_file:
#with open(path+'metadata.csv') as csv_file:
with open(path + 'train.csv') as csv_file:

    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
#        print(row)
        if line_count == 0:
            line_count += 1
        else:
#            if line_count == 3:
#                break
            load_tags.update({row[5]: row[4]})
            #print(load_tags)
            line_count += 1
#exit()
counter = 0
for folder in listdir(path + 'real_vs_fake/real-vs-fake/train/'):
#    if counter > 15:
#        break
    if folder == '.DS_Store' or folder == 'metadata.csv':
        continue
#    for filename in listdir('../datasets/faces-sample/' + folder):
    isReal = folder == 'real'

    print("Currently processing: " + folder)
    for filename in listdir(path + 'real_vs_fake/real-vs-fake/train/' + folder):
        curpath = path +'real_vs_fake/real-vs-fake/train/' +  folder + '/'
#        print(curpath)
        # load image
        img_data = image.imread(path +'real_vs_fake/real-vs-fake/train/' +  folder + '/' + filename)
        # store loaded image
        loaded_images.append(img_data)
        #print(len(load_tags))
        #print(load_tags.get('test/fake/3986S7FE80.jpg')) 
        if (load_tags['train/' + folder + '/' + filename] == ('real' or '1' or 1)): 
#             print('is real')
             l = [[1, 0]] * (len(loaded_images) - len(y))
             y.extend(l)
        else:
             #print('is fake')
             l = [[0, 1]] * (len(loaded_images) - len(y))
             y.extend(l)
        y+= [load_tags['train/'+folder+'/'+filename]] * (len(loaded_images) - len(y))
        counter += 1
        #break
#exit()
Y_train = np.asarray(y)
loaded_images= np.asarray(loaded_images)
print(loaded_images.shape)
chdir('/data/traindata')
partition = len(loaded_images)//4
X_train = loaded_images
print('saving quarters of train set')
np.savez_compressed('x_train1', X_train[:partition])
np.savez_compressed('y_train1', Y_train[:partition])
np.savez_compressed('x_train2', X_train[partition + 1:partition*2])
np.savez_compressed('y_train2', Y_train[partition + 1 : partition*2])
np.savez_compressed('x_train3', X_train[partition*2 + 1:partition*3])
np.savez_compressed('y_train3', Y_train[partition*2 + 1:partition*3])
np.savez_compressed('x_train4', X_train[partition*3 + 1:partition*4])
np.savez_compressed('y_train4', Y_train[partition*3 + 1:partition*4])
print('normalizing RGB of each quarter and saving normalized quarters')
X_train = np.load('x_train1.npz')['arr_0']/255.
np.savez_compressed('x_train1', X_train)
X_train = np.load('x_train2.npz')['arr_0']/255.
np.savez_compressed('x_train2', X_train)
X_train = np.load('x_train3.npz')['arr_0']/255.
np.savez_compressed('x_train3', X_train)
X_train = np.load('x_train4.npz')['arr_0']/255.
np.savez_compressed('x_train4', X_train)
