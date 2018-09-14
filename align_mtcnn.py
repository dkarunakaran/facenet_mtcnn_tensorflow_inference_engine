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
    def __init__(self, data_path=None, model_dir='mtcnn_model/all_in_one', threshold=[0.8, 0.8, 0.8], factor=0.7, minsize = 20, margin=44, image_size=182, detect_multiple_faces=False, aligned_dataset=None):
        self.data_path = data_path
        self.model_dir = model_dir
        self.threshold = threshold
        self.factor = factor
        self.minsize = minsize
        self.margin = margin
        self.image_size = image_size
        self.detect_multiple_faces = detect_multiple_faces
        self.aligned_dataset = aligned_dataset

    def get_bounding_boxes(self, single=False, img_path=None):
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
                    if not single:
                        for filename in os.listdir(self.data_path):
                            img = cv2.imread(self.data_path+"/"+filename, 1)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            start_time = time.time()
                            bounding_boxes, points = detect_face(img, self.minsize,
                                                                pnet_fun, rnet_fun, onet_fun,
                                                                self.threshold, self.factor)
                            self.process_bounding_boxes(img, bounding_boxes, nrof_successfully_aligned, filename)
                            
                    else:
                        img = cv2.imread(img_path, 1)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        start_time = time.time()
                        bounding_boxes, points = detect_face(img, self.minsize,
                                                            pnet_fun, rnet_fun, onet_fun,
                                                            self.threshold, self.factor)
                        #print(points)
                        self.process_bounding_boxes(img, bounding_boxes, nrof_successfully_aligned, img_path)

    def process_bounding_boxes(self, img, bounding_boxes, nrof_successfully_aligned, filename):
        nrof_faces = bounding_boxes.shape[0]
        print("No. of faces detected: {}".format(nrof_faces))

        if nrof_faces>0:
            det = bounding_boxes[:,0:4]
            det_arr = []
            img_size = np.asarray(img.shape)[0:2]
            if nrof_faces>1:
                if self.detect_multiple_faces:
                    for i in range(nrof_faces):
                        det_arr.append(np.squeeze(det[i]))
                else:
                    bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                    img_center = img_size / 2
                    offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                    offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                    index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                    det_arr.append(det[index,:])
            else:
                det_arr.append(np.squeeze(det))

            for i, det in enumerate(det_arr):
                det = np.squeeze(det)
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0]-self.margin/2, 0)
                bb[1] = np.maximum(det[1]-self.margin/2, 0)
                bb[2] = np.minimum(det[2]+self.margin/2, img_size[1])
                bb[3] = np.minimum(det[3]+self.margin/2, img_size[0])
                cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                scaled = misc.imresize(cropped, (self.image_size, self.image_size), interp='bilinear')
                nrof_successfully_aligned += 1
                filename_base, file_extension = os.path.splitext(filename)
                if self.detect_multiple_faces:
                    output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                else:
                    output_filename_n = "{}{}".format(filename_base, file_extension)
                misc.imsave(self.aligned_dataset+"/"+output_filename_n, scaled)
                #text_file.write('%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))
        else:
            print('Unable to align "%s"' % image_path)
            #text_file.write('%s\n' % (output_filename))


if __name__ == '__main__':
    alignMTCNN = AlignMTCNN('people', aligned_dataset='aligned_dataset')
    alignMTCNN.get_bounding_boxes()
