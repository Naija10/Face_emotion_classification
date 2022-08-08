import numpy as np
import cv2
import os
import random

parent="Faceimages/"

folders=os.listdir(parent)

classes=["Angry","Happy","Normal","Sad","Surprised"]

dataset=[]

for folder in folders:
     full_path=os.path.join(parent,folder)
     target=classes.index(folder)
     imgs=os.listdir(full_path)
     for img in imgs:
          try:
               img_path=os.path.join(full_path,img)
               img=cv2.imread(img_path)
               resized_img=cv2.resize(img,(50,50))  # resizing the image to prevent computational complexity
               dataset.append([resized_img,target])
          except:
               print("error")
print(dataset)
random.shuffle(dataset)    # shuffling the dataset to prevent overfitting
dataset=np.array(dataset)
print(dataset.shape)
np.save("Datasets/TRAINofemotions.npy",dataset)

          
