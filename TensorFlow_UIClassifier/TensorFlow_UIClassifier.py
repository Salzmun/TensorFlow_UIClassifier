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

#from keras.datasets import fashion_mnist

#(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

filename = "C:/Users/Max/source/repos/TensorFlow_UIClassifier/TensorFlow_UIClassifier/TFRecords/train-000.tfrecords"
filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)

for raw_record in raw_dataset.take(10):
    print(repr(raw_dataset))

(x_train, y_train), (x_test, y_test) = raw_dataset

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
    model.fit(x_train, y_train, epochs=epoch)
    model.evaluate(x_test,  y_test, verbose=2)
    
######### setup #########
def main():
    tensorflowmagic()

if __name__ == "__main__":
    main()