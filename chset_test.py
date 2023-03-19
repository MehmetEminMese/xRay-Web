# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 21:15:41 2022

@author: casper
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import warnings
warnings.filterwarnings("ignore")
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets,layers,models
import os
class_names=['normal','pheumonia']
#model=models.load_model('image_classifier_eskipath.model')
model=models.load_model('C://Users//casper//Desktop//chest_xray//image_classifier.model')
#model=models.load_model('image_classifier_2.model')
# #img=cv.imread('C://Users//casper//Desktop//dataset//dataset_32//NORMAL//NORMAL2-IM-1442-0001.jpg')
# img=cv.imread('C://Users//casper//Desktop//dataset//dataset_32//PNEUMONIA//person1946_bacteria_4875.jpg')
# #img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
# plt.imshow(img,cmap=plt.cm.binary)
# prediction=model.predict(np.array([img])/255)
# index=np.argmax(prediction)
# print(f'Prediction is {class_names[index]}')
# plt.show()

# ("C://Users//casper//Desktop//chest_xray//test_split//PNEUMONIA//{images}".format(images=image))

count_hasta=0
count_normal=0
# get the path/directory
folder_dir = "C://Users//casper//Desktop//chest_xray//dataset_32_2//NORMAL"
for image in os.listdir(folder_dir):
#check if the image ends with png
    if (image.endswith(".jpeg")):
        #test_img=load_img(images,target_size=(128,128))
        img=cv.imread(r"C://Users//casper//Desktop//chest_xray//dataset_32_2//NORMAL//{images}".format(images=image))
        #test_img=load_img(images)
        # test_img = img_to_array(img)
        # test_img = test_img/255
        # test_img=np.expand_dims(test_img,axis=0)
        #result = predict_model.predict(test_img)
        prediction=model.predict(np.array([img])/255)
        result=model.predict(np.array([img])/255)
        index=np.argmax(prediction)
        print(f'Prediction is {class_names[index]}')
        if(class_names[index]=='normal'):
            count_normal+=1
        else:
            count_hasta+=1
#img=cv.imread(r"C://Users//casper//Desktop//chest_xray//test_split//NORMAL//NORMAL2-IM-0686-0001.jpeg")
#print("shape",img.shape)
print("count_hasta",count_hasta)
print("count_normal",count_normal)
global json
json={
    "hasta":count_hasta,
    "normal":count_normal
}
def return_Json(json):
    return json
print(return_Json(json))


        # if(result[0][0]>0.5):
        #     print(value," hasta")
        #     print(result[0][0])
        #     count_hasta +=1
        # else:
        #     print(value, " normal")
        #     print(result[0][0])
        #     count_normal +=1
