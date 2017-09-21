#! /usr/bin/python3
# -*- coding : UTF-8 -*-


from skimage import data, transform
import os
import numpy as np
from skimage.color import rgb2gray
import tensorflow as tf

#set constant
batch_size = 10
epoch = 200
learning_rate = 0.1


def load_data(data_directory):
    directories = [ d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d)) ]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [ os.path.join(label_directory, f) for f in os.listdir(label_directory) if f.endswith(".ppm") ]
    
        for f in file_names:
            images.append(data.imread(f))
            labels.append(int(d))
    
    return images, labels

ROOT_PATH = "/home/liuchang/pythonlearn"
train_data_directory = os.path.join(ROOT_PATH, "trafficSign/trafficSignData/Testing")
test_data_directory = os.path.join(ROOT_PATH, "trafficSign/trafficSignData/Training")
images, labels = load_data(train_data_directory)

images = np.array(images)
labels = np.array(labels)

#Rescale the images in the 'images' array
images28 = [transform.resize(image, (28, 28)) for image in images]

#Convert 'images28' to an array
images28 = np.array(images28)

#Convert 'images28' to grayscale
images28 = np.array([rgb2gray(image) for image in images28])
#print (images28.shape)
#Initialize placeholders
x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28])
y = tf.placeholder(dtype=tf.int32, shape=[None])

W = tf.Variable(tf.truncated_normal([3, 3, 1, 32], stddev = 0.1))  #obtian a zheng tai fen bu shu ju  and average value float32
#print (W)
#print (x)
#W = tf.Variable([3, 3, 1, 32])
X = tf.reshape(x,[-1, 28, 28, 1])
#x = tf.expand_dims(x, -1)
#print (x)
#Cnn to convert
x_conv = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')
#print (x_conv.getshape())
#pirnt (x_conv)
#Pooling the data
x_pool = tf.nn.avg_pool(x_conv,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

x_conv1 = tf.nn.conv2d(x_pool, W, strides=[1, 1, 1, 1], padding='SAME')
x_pool1 = tf.nn.avg_pool(x_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#Flatten the input data
images_flat = tf.contrib.layers.flatten(x_pool1)

#Fully connected layer
logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

#Define a loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))

#Define an optimizer
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)


''''''
#Convert logits to label indexes
correct_pred = tf.argmax(logits, 1)

#Define an accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#Runing the neural network
tf.set_random_seed(1234)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epoch):
        _, loss_value = sess.run([train_op, loss], feed_dict={x:images28, y:labels})
        if i%10 == 0:
            print("loss: ", loss_value)
  #      print("DONE A EPOCH!")
            #print("accuracy: ", train_op.shape)
            print("!!!!!")
    '''
    Evaluating Your noural network
    '''   
    # downloading your testing data
    test_images, test_labels = load_data(test_data_directory)
    #print(np.array(test_images)[0].size)
    #deal with the image
    test_images28 = [transform.resize(image, (28, 28)) for image in test_images]

    #Convert the image color to gray
    test_images28 = np.array([rgb2gray(image) for image in test_images28])
    predicted = sess.run([correct_pred], feed_dict={x:test_images28})[0]
    match_mount = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])
    accuracy = match_mount / len(test_labels)
    print("Accuracy: {:.3f}".format(accuracy))
