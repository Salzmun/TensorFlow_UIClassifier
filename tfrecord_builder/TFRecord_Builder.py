##### Imports #####
import tensorflow as tf
import numpy as np
import os
import sys
import cv2
from random import shuffle

#### Functions ####

def getFiles(filePath,switcheroni):
    """Returns Arrays with Fileanmes and Labels for Train, Validation and Test.
    
    Keyword arguments:
    filePath -- Path to folder/files, which should be turned into TFRecords
    """
    addrs = [name for name in os.listdir(filePath) if name.endswith(".jpg")]
    if len(addrs) == 0: 
        print('Could not find *.jpg files in provided directory.') 
    else:
        print('%4d Files found' % (len(addrs)))
    labels = GetLabels(addrs)
    if switcheroni:
        c = list(zip(addrs, labels))
        shuffle(c)
        addrs, labels = zip(*c)

    train_addrs = addrs[0:int(0.6*len(addrs))]
    train_labels = labels[0:int(0.6*len(labels))]
    val_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addrs))]
    val_labels = labels[int(0.6*len(addrs)):int(0.8*len(addrs))]
    test_addrs = addrs[int(0.8*len(addrs)):]
    test_labels = labels[int(0.8*len(labels)):]
    return train_addrs,train_labels,val_addrs,val_labels,test_addrs,test_labels 


def GetLabels(addrs):
    """Genetrates an Array with labes corresponding to the given Array. 

    Keyword arguments:
    addrs -- Array with paths to process
    """
    labels = []
    for filename in addrs:
        if 'box' in filename:
            labels.append(0)
        elif 'text' in filename:
            labels.append(1)
        elif'icon' in filename:
            labels.append(2)
    return labels

def load_image(addr,size_x,size_y):
    """Reads the Image from the given Address, will greyscale it and lastly scales it down to size_x by size_y.

    Keyword arguments:
    addr -- Path of the Image
    size_x -- IntValue for resize/scaling of picture
    size_y -- IntValue for resize/scaling of picture
    """
    img = cv2.imread(addr)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (50, 50),interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32)
    return img

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def createRecord(recPath, filePath, filename, file_array, label_array,size_x,size_y,name):
    """Builds a TFRecord out of the given Arrays. Will call load_image for processing jpgs.

    Keyword arguments:
    recPath -- Where to Save the Record
    filename -- Name of the TFRecord
    file_array -- Array with Filenames
    label_array -- Array wit corresponding Labels
    size_x -- target x-axis lenght of given pictures
    size_y -- target y-axis length of given pictures
    """
    writer = tf.io.TFRecordWriter(os.path.join(recPath, filename))
    for i in range(len(file_array)):
        if not i % 1000:
            print ('Train data: {}/{}'.format(i, len(file_array)))
            sys.stdout.flush()
        img = load_image(os.path.join(filePath, file_array[i]),size_x,size_y)
        label = label_array[i]
        feature = {name +'/label': _int64_feature(label),
                   name + '/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    writer.close()
    sys.stdout.flush()

##### Staging #####

def main():
    ##### Variables #####
    shuffle_data = True
    filePath = 'C:/Users/Max/source/repos/TensorFlow_UIClassifier/TFRecord_Builder/TrainingData_rough/'
    recordPath = "C:/Users/Max/source/repos/TensorFlow_UIClassifier/TFRecord_Builder/Output/"

    train_filename = 'train.tfrecords'
    val_filename = 'val.tfrecords'
    test_filename = 'test.tfrecords'

    img_x = 50
    img_y = 50

    ### End Variables ###

    train_file,train_label,val_file,val_label,test_file,test_label = getFiles(filePath,shuffle_data)
    createRecord(recordPath, filePath, train_filename, train_file, train_label, img_x, img_y, 'train')
    createRecord(recordPath, filePath, val_filename, val_file, val_label, img_x,img_y,'val')
    createRecord(recordPath, filePath, test_filename, test_file, test_label, img_x,img_y,'test')


if __name__ == "__main__":
    main()