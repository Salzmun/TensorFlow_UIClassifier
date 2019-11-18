import glob
import os
import sys
import tensorflow as tf
import cv2
from random import shuffle
import numpy as np


shuffle_data = True  # shuffle the addresses before saving
train_path = 'C:/Users/Max/source/repos/TensorFlow_UIClassifier/TFRecord_Builder/TrainingData_rough/'
print(train_path)

# read addresses and labels from the 'train' folder

addrs = [name for name in os.listdir(train_path) if name.endswith(".jpg")]
#check if loaded correctly
if len(addrs) == 0: 
    print('Could not find *.jpg files in provided directory.')
for x in addrs:
    print(x)


#labels = [0 if 'box' in addr else 1 for 'icon' in addr else 2 for 'text' in addrs]  
labels = []
for filename in addrs:
    if 'box' in filename:
        labels.append(0)
    elif 'text' in filename:
        labels.append(1)
    elif'icon' in filename:
        labels.append(2)

#check if loaded correctly
for x in labels:
    print(x)


# to shuffle data
if shuffle_data:
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)
    
# Divide the hata into 60% train, 20% validation, and 20% test
train_addrs = addrs[0:int(0.6*len(addrs))]
train_labels = labels[0:int(0.6*len(labels))]
val_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addrs))]
val_labels = labels[int(0.6*len(addrs)):int(0.8*len(addrs))]
test_addrs = addrs[int(0.8*len(addrs)):]
test_labels = labels[int(0.8*len(labels)):]

#Getting the labels from filenames 
#   NOT IN USE AS OFF NOW
def GetLabels(addrs):
    labels = []
    for filename in addrs:
        if 'box' in filename:
            labels.append()
        elif 'text' in filename:
            labels.append()
        elif'icon' in filename:
            labels.append()
    return labels

def load_image(addr):
    # read an image and resize to (50, 50)
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)
    #img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.resize(img, (50, 50),interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

recordPath = "C:/Users/Max/source/repos/TensorFlow_UIClassifier/TFRecord_Builder/Output/"

train_filename = 'train.tfrecords'  # address to save the TFRecords file
# open the TFRecords file
writer = tf.io.TFRecordWriter(os.path.join(recordPath, train_filename))
for i in range(len(train_addrs)):
    # print how many images are saved every 1000 images
    if not i % 1000:
        print ('Train data: {}/{}'.format(i, len(train_addrs)))
        sys.stdout.flush()
    # Load the image
    img = load_image(os.path.join(train_path, train_addrs[i]))
    label = train_labels[i]
    # Create a feature
    feature = {'train/label': _int64_feature(label),
               'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())
    
writer.close()
sys.stdout.flush()



# open the TFRecords file
val_filename = 'val.tfrecords'  # address to save the TFRecords file
writer = tf.io.TFRecordWriter(os.path.join(recordPath,val_filename))
for i in range(len(val_addrs)):
    # print how many images are saved every 1000 images
    if not i % 1000:
        print ('Val data: {}/{}'.format(i, len(val_addrs)))
        sys.stdout.flush()
    # Load the image
    img = load_image(os.path.join(train_path, val_addrs[i]))
    label = val_labels[i]
    # Create a feature
    feature = {'val/label': _int64_feature(label),
               'val/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())
writer.close()
sys.stdout.flush()

# open the TFRecords file
test_filename = 'C:/Users/Max/source/repos/TensorFlow_UIClassifier/TFRecord_Builder/Output/test.tfrecords'  # address to save the TFRecords file
writer = tf.io.TFRecordWriter(os.path.join(recordPath,test_filename))
for i in range(len(test_addrs)):
    # print how many images are saved every 1000 images
    if not i % 1000:
        print ('Test data: {}/{}'.format(i, len(test_addrs)))
        sys.stdout.flush()
    # Load the image
    img = load_image(os.path.join(train_path, test_addrs[i]))
    label = test_labels[i]
    # Create a feature
    feature = {'test/label': _int64_feature(label),
               'test/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())
writer.close()
sys.stdout.flush()


#def main():
    


#if __name__ == "__main__":
#    main()