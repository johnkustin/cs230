import tensorflow as tf
import argparse
import inceptionv4utils as utils
import numpy as np
# import wandb

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

parser = argparse.ArgumentParser()
parser.add_argument('--nets', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=100)
args = parser.parse_args()

print(args)
# wandb.init(project="conv-nets", name=args.nets.lower())

model = utils.choose_nets(args.nets)
cifar100 = tf.keras.datasets.cifar100

(x_train,y_train),(x_test,y_test)=cifar100.load_data()
x_train,x_test=x_train/255.0,x_test/255.0

# add train and test here
"""
(x_train_ptr, y_train_ptr) = np.load('/data/testdata/x_dataset_half.npz'), np.load('/data/testdata/y_dataset_half.npz')
x_train = x_train_ptr['arr_0']
y_train = y_train_ptr['arr_0']
x_train_ptr.close()
y_train_ptr.close()
numimages = x_train.shape[0]
(x_test, y_test) = x_train[:250,:,:,:],y_train[:250,:]
numTestImages = x_test.shape[0]
x_train = x_train[251:,:,:,:]
y_train = y_train[251:,:]
numTrainImages = x_train.shape[0]
# flattens train data
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
print('x train shape ') 
print(x_train.shape)
print('y train shape ') 
print(y_train.shape)
print('x test shape ' ) 
print(x_test.shape)
print('y test shape ' ) 
print(y_test.shape)

newY = []
for data in y_train:
    if data[0] == 1: #real
        newY.append(1)
    else: #fake
        newY.append(0)
y_train = newY

newYT = []
for data in y_test:
    if data[0] == 1: #real
        newYT.append(1)
    else: #fake
        newYT.append(0)
y_test = newYT

x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
print('xtrain tensor shape')
print(tf.shape(x_train).numpy())
y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
print('ytrain tensor shape')
print(tf.shape(y_train).numpy())
x_train = tf.reshape(x_train, [numTrainImages, 256, 256, 3])
print('xtrain reshaped tensor shape')
print(tf.shape(x_train).numpy())

x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
x_test = tf.reshape(x_test, [numTestImages, 256, 256, 3])
y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)
"""
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(args.batch_size)
test_ds = tf.data.Dataset.from_tensor_slices(
    (x_test, y_test)).batch(args.batch_size)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(args.lr)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='test_accuracy')


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    print(loss)
    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

print('starting model')
for epoch in range(args.epochs):
    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    template = 'Epoch: [{}/{}], Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1,
                          args.epochs,
                          train_loss.result(),
                          train_accuracy.result()*100,
                          test_loss.result(),
                          test_accuracy.result()*100))
    # wandb.log({
    #     "TrainLoss": train_loss.result(),
    #     "TestLoss": test_loss.result(),
    #     "TrainAcc": train_accuracy.result()*100,
    #     "TestAcc": test_accuracy.result()*100
    # })
    train_loss.reset_states()
    test_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()
