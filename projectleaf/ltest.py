# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 09:00:22 2018

@author: Aditya



# -*- coding: utf-8 -*-
"""


from __future__ import print_function
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input
import numpy as np
import os
import json
import pickle
import cv2
with open('C:/Users/Adity/Desktop/projectleaf/conf/conf.json') as f:    
  config = json.load(f)
model_name 		= config["model"]
weights 		= config["weights"]
include_top 	= config["include_top"]
train_path 		= config["train_path"]
test_path 		= config["test_path"]
features_path 	= config["features_path"]
labels_path 	= config["labels_path"]
test_size 		= config["test_size"]
results 		= config["results"]
model_path 		= config["model_path"]
seed 			= config["seed"]
classifier_path = config["classifier_path"]
print ("[INFO] loading the classifier...")
classifier = pickle.load(open(classifier_path, 'rb'))
if model_name == "vgg16":
	base_model = VGG16(weights=weights)
	model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)
	image_size = (224, 224)
else:
    model_name == "inceptionv3"
	

    input_tensor = Input(shape=(224, 224, 3))

    model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=True)

train_labels = os.listdir(train_path)
test_images = os.listdir(test_path)
for image_path in test_images:
	path 		= test_path + "/" + image_path
	img 		= image.load_img(path, target_size=(224,224))
	x 			= image.img_to_array(img)
	x 			= np.expand_dims(x, axis=0)
	x 			= preprocess_input(x)
	feature 	= model.predict(x)
	flat 		= feature.flatten()
	flat 		= np.expand_dims(flat, axis=0)
	preds 		= classifier.predict(flat)
	prediction 	= train_labels[preds[0]]
	print ("It is a " + train_labels[preds[0]])
	img_color = cv2.imread(path, 1)
	cv2.putText(img_color, "It is a " + prediction, (100,100), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,0,0), 2)
	cv2.imshow("test", img_color)
	key = cv2.waitKey(0) & 0xFF
	if (key == ord('q')):
		cv2.destroyAllWindows()
	
 