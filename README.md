# FaceNet and MTCNN  using Tensorflow

### Required packages
* Python 3.6
* Tensorflow 1.8
* opencv 3.4
* Scipy
* Numpy
* Pickle
* Sklearn

### Download the pretrained weights
* Download [this](https://drive.google.com/file/d/1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-/view) pretrained Facenet model and copy to model folder.
* Download [this](https://github.com/wangbm/MTCNN-Tensorflow/tree/master/save_model) pretrained MTCNN models and copy to mtcnn_model.

### Steps to create the embeddings
* Add images to 'people' folder for creating the embeddings
* Run the below code to create the embeddings
```
face_embedding = FaceEmbedding()
embedding = face_embedding.convert_to_embedding()
```
### Steps the comapare nw images with existing embedding created by above step:
* Run the below code for comparison(Please note provide the image to compare in convert_to_embedding method).
```
face_embedding = FaceEmbedding()
embedding = face_embedding.convert_to_embedding(single=True, img_path='face6.jpg')
emb_list = face_embedding.load_pickle()
face_embedding.ecuclidean_distance(emb_list, embedding)
```

### Custom training for Facnet and MTCNN models
This repos is mainly make use of pretrained weights and act as inference engine for face recognition. If you want to train, facnet and MTCNN models for further, Use these great two links. Once the training finished, you can copy back the models to this repo and can start doing inference.

* Facenet training: https://github.com/davidsandberg/facenet/wiki
* MTCNN training: https://github.com/wangbm/MTCNN-Tensorflow
