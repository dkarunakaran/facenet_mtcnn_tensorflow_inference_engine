# Compute the 128D vector that describes the face in img identified by
# shape.  In general, if two face descriptor vectors have a Euclidean
# distance between them less than 0.6 then they are from the same
# person, otherwise they are from different people. 
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import tqdm
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
import numpy as np
import pickle
import cv2
from align_mtcnn import AlignMTCNN

class FaceEmbedding:
    
    def __init__(self, data_path='people', model_dir='model/20180402-114759'):
        self.data_path = data_path
        self.model_dir = model_dir
        self.alignMTCNN = AlignMTCNN()
        self.images_placeholder = None
        self.embeddings = None
        self.phase_train_placeholder=None
        self.embedding_size=None
        self.image_size = 160
        self.threshold=[0.8, 0.8, 0.8]
        self.factor=0.7
        self.minsize = 20
        self.margin=44
        self.detect_multiple_faces = False


    def convert_to_embedding(self, single=False, img_path=None):
        extracted = []
        with tf.Graph().as_default():
                with tf.Session() as sess:
                    self.sess = sess
                    # Load the model
                    facenet.load_model(self.model_dir)
                    
                    # Get input and output tensors
                    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                    self.images_placeholder = tf.image.resize_images(images_placeholder,(self.image_size, self.image_size))
                    self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                    self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                    self.embedding_size = self.embeddings.get_shape()[1]
                    if not single:
                        for filename in os.listdir(self.data_path):
                            img = cv2.imread(self.data_path+"/"+filename, 1)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            bounding_boxes, points = self.alignMTCNN.get_bounding_boxes(image=img)
                            faces = self.get_faces(img, bounding_boxes, points, filename)
                            extracted.append(faces)
                        with open('extracted_embeddings.pickle','wb') as f:
                            pickle.dump(extracted,f)
                        return None
                    else:
                        img = cv2.imread(img_path, 1)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        bounding_boxes, points = self.alignMTCNN.get_bounding_boxes(image=img)
                        faces = self.get_faces(img, bounding_boxes, points, img_path)
                        return faces
    
    def get_faces(self, img, bounding_boxes, points, filename):
        faces = []
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
                resized = cv2.resize(cropped, (self.image_size, self.image_size),interpolation=cv2.INTER_CUBIC)
                prewhitened = facenet.prewhiten(resized)
                faces.append({'name': filename,'rect':[bb[0],bb[1],bb[2],bb[3]],'embedding':self.get_embedding(prewhitened)})

        return faces

    def get_embedding(self, processed_img):
        reshaped = processed_img.reshape(-1, self.image_size, self.image_size, 3)
        feed_dict = {self.images_placeholder:reshaped, self.phase_train_placeholder:False }
        feature_vector = self.sess.run(self.embeddings, feed_dict=feed_dict)
        return feature_vector

    
    def ecuclidean_distance(self, emb_list, embedding):
        for emb in emb_list:
            dist = np.sqrt(np.sum(np.square(np.subtract(emb[0]['embedding'], embedding[0]['embedding']))))
            print(emb[0]['name'])
            print('  %1.4f  ' % dist, end='')
            print("\n")

    def load_pickle(self):
        embeddings = pickle.load( open( "extracted_embeddings.pickle", "rb" ) )
        return embeddings

    

if __name__ == '__main__':
    face_embedding = FaceEmbedding()
    embedding = face_embedding.convert_to_embedding(single=True, img_path='face6.jpg')
    #embedding = face_embedding.convert_to_embedding()
    emb_list = face_embedding.load_pickle()
    face_embedding.ecuclidean_distance(emb_list, embedding)

    