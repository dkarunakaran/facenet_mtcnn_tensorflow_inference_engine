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
    
    def __init__(self, aligned_dataset='aligned_dataset', model_dir='model/20180402-114759'):
        self.aligned_dataset = aligned_dataset
        if not os.path.exists(self.aligned_dataset):
            os.makedirs(self.aligned_dataset)
        self.model_dir = model_dir
        self.alignMTCNN = AlignMTCNN(aligned_dataset=self.aligned_dataset)



    def convert_to_embedding(self, single=False, img_path=None):
        extracted_dict = {}
        image_size = 160
        with tf.Graph().as_default():
                with tf.Session() as sess:

                    # Load the model
                    facenet.load_model(self.model_dir)
                    
                    # Get input and output tensors
                    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                    images_placeholder = tf.image.resize_images(images_placeholder,(image_size, image_size))
                    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                    embedding_size = embeddings.get_shape()[1]
                    if not single:
                        for filename in os.listdir(self.aligned_dataset):
                            images = facenet.load_image(self.aligned_dataset+"/"+filename, False, False, image_size)
                            feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                            feature_vector = sess.run(embeddings, feed_dict=feed_dict)
                            extracted_dict[filename] =  feature_vector
                        with open('extracted_dict.pickle','wb') as f:
                            pickle.dump(extracted_dict,f)
                        return None
                    else:
                        self.alignMTCNN.get_bounding_boxes(single=True, img_path=img_path)
                        images = facenet.load_image(self.aligned_dataset+"/"+img_path, False, False, image_size)
                        feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                        feature_vector = sess.run(embeddings, feed_dict=feed_dict)
                        return feature_vector

    def ecuclidean_distance(self, emb_list, embedding):
        for emb in emb_list:
            result = np.sum(np.square(embedding-emb_list.get(emb)))
            dist = np.sqrt(np.sum(np.square(np.subtract(emb_list.get(emb), embedding))))
            print(emb)
            print('  %1.4f  ' % dist, end='')
            print("\n")

    def load_pickle(self):
        embeddings = pickle.load( open( "extracted_dict.pickle", "rb" ) )
        return embeddings

    

if __name__ == '__main__':
    face_embedding = FaceEmbedding(aligned_dataset='temp')
    embedding = face_embedding.convert_to_embedding(single=True, img_path='face4.jpg')
    #embedding = face_embedding.convert_to_embedding('face1.png', True)
    emb_list = face_embedding.load_pickle()
    face_embedding.ecuclidean_distance(emb_list, embedding)

    