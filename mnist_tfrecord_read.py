from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time
from PIL import Image

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import mnist

FLAGS = None

TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'

tfrecords_filename='/tmp/train.tfrecords'
filename_queue = tf.train.string_input_producer([tfrecords_filename],)
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue) 
features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'image_raw' : tf.FixedLenFeature([], tf.string),
                                       }) 

image = tf.decode_raw(features['image_raw'], tf.uint8)
image = tf.reshape(image, [28,28])
print("image:", image)

label = tf.cast(features['label'], tf.int64)
print("label:", label)

with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(20):
        example, l = sess.run([image, label])
        img = Image.fromarray(example)
        img.save('./'+str(i)+'_''Label_'+str(l)+'.jpg')
        #print(example, l)
    coord.request_stop()
    coord.join(threads)
