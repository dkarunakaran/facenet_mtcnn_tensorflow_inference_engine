# MIT License
#
# Copyright (c) 2017 Baoming Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import argparse
import time
from scipy import misc
import tensorflow as tf
import cv2
import numpy as np
import os
from align_mtcnn_src.mtcnn import PNet, RNet, ONet
from tools import detect_face, get_model_filenames

class AlignMTCNN:
    def __init__(self, model_dir='mtcnn_model/all_in_one', threshold=[0.8, 0.8, 0.8], factor=0.7, minsize = 20, margin=44, image_size=182, detect_multiple_faces=False):
        self.model_dir = model_dir
        self.threshold = threshold
        self.factor = factor
        self.minsize = minsize
        self.margin = margin
        self.image_size = image_size
        self.detect_multiple_faces = detect_multiple_faces

    def get_bounding_boxes(self, image=None):
        with tf.device('/gpu:0'):
            with tf.Graph().as_default():
                config = tf.ConfigProto(allow_soft_placement=True)
                with tf.Session(config=config) as sess:
                    nrof_images_total = 0
                    nrof_successfully_aligned = 0
                    file_paths = get_model_filenames(self.model_dir)
                    
                    #if the Pnet, Rnet, and Onet trained separetely
                    if len(file_paths) == 3:
                        image_pnet = tf.placeholder(
                            tf.float32, [None, None, None, 3])
                        pnet = PNet({'data': image_pnet}, mode='test')
                        out_tensor_pnet = pnet.get_all_output()

                        image_rnet = tf.placeholder(tf.float32, [None, 24, 24, 3])
                        rnet = RNet({'data': image_rnet}, mode='test')
                        out_tensor_rnet = rnet.get_all_output()

                        image_onet = tf.placeholder(tf.float32, [None, 48, 48, 3])
                        onet = ONet({'data': image_onet}, mode='test')
                        out_tensor_onet = onet.get_all_output()

                        saver_pnet = tf.train.Saver(
                                        [v for v in tf.global_variables()
                                            if v.name[0:5] == "pnet/"])
                        saver_rnet = tf.train.Saver(
                                        [v for v in tf.global_variables()
                                            if v.name[0:5] == "rnet/"])
                        saver_onet = tf.train.Saver(
                                        [v for v in tf.global_variables()
                                            if v.name[0:5] == "onet/"])

                        saver_pnet.restore(sess, file_paths[0])

                        def pnet_fun(img): return sess.run(
                            out_tensor_pnet, feed_dict={image_pnet: img})

                        saver_rnet.restore(sess, file_paths[1])

                        def rnet_fun(img): return sess.run(
                            out_tensor_rnet, feed_dict={image_rnet: img})

                        saver_onet.restore(sess, file_paths[2])

                        def onet_fun(img): return sess.run(
                            out_tensor_onet, feed_dict={image_onet: img})

                    else:
                        saver = tf.train.import_meta_graph(file_paths[0])
                        saver.restore(sess, file_paths[1])

                        def pnet_fun(img): return sess.run(
                            ('softmax/Reshape_1:0',
                                'pnet/conv4-2/BiasAdd:0'),
                            feed_dict={
                                'Placeholder:0': img})

                        def rnet_fun(img): return sess.run(
                            ('softmax_1/softmax:0',
                                'rnet/conv5-2/rnet/conv5-2:0'),
                            feed_dict={
                                'Placeholder_1:0': img})

                        def onet_fun(img): return sess.run(
                            ('softmax_2/softmax:0',
                                'onet/conv6-2/onet/conv6-2:0',
                                'onet/conv6-3/onet/conv6-3:0'),
                            feed_dict={
                                'Placeholder_2:0': img})
                    
                    bounding_boxes, points = detect_face(image, self.minsize,
                                                            pnet_fun, rnet_fun, onet_fun,
                                                            self.threshold, self.factor)
                    return bounding_boxes, points

if __name__ == '__main__':
    alignMTCNN = AlignMTCNN('people')
    alignMTCNN.get_bounding_boxes()
