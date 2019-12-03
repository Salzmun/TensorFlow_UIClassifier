######### Imports #########
from __future__ import absolute_import, division, print_function, unicode_literals


import functools
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras


######### Parameters #########

epoch = 5
inputx = 28
inputy = 28

########## Generating Data #########



######### Data Binding #########



######### Setting Datasource #########

from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#file_path = "C:/Users/Max/source/repos/TensorFlow_UIClassifier/TensorFlow_UIClassifier/TFRecords/"
#train_filename = "train.tfrecords"
#train_filenames = [train_filename]
#test_filename = "test.tfrecords"
#test_filenames = [test_filename]
#val_filename = "val.tfrecords"
#val_filenames = [val_filename]

#train_dataset = tf.data.TFRecordDataset(train_filenames)

#test_dataset = tf.data.TFRecordDataset(test_filenames)

#val_dataset = tf.data.TFRecordDataset(val_filenames)

#for raw_record in raw_dataset.take(10):
#    print(repr(raw_dataset))

#(x_train, y_train), (x_test, y_test) = raw_dataset


def getlabels(j):
    if j == 0:
        return 'box'
    elif j == 1:
        return 'text'
    elif j == 2:
        return 'icon'
    else:
        return 'derp'

########## Model Building #########
def tensorflowmagic():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(inputx, inputy)),
        tf.keras.layers.Dense(128, activation='sigmoid'),
        tf.keras.layers.Dense(128, activation='sigmoid'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='sgd',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    # Error when checking input: expected flatten_input to have 3 dimensions, but got array with shape ()
    # IMGs in record already flat?? 
    # 1.) Change Inputlayer to utilize flattened / serialized images
    # 2.) Read strings from record an rebuild them as images...
    model.fit(train_dataset, epochs=epoch)
    model.evaluate(val_dataset, verbose=2)  
    #model.fit(x_train, y_train, epochs=epoch)
    #model.evaluate(x_test,  y_test, verbose=2)
    

######### setup #########
def main():
    tensorflowmagic()

if __name__ == "__main__":
    main()