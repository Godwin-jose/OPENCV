import os
from sklearn.tree import DecisionTreeClassifier
import cv2 as cv
import numpy as np
import joblib

path="F:\\model\\train"
list=[]
feactures=[]
labels=[]
files=os.listdir(path)
x=-1
for each in files:
    way=os.path.join(path,each)
    files=os.listdir(way)
    x+=1
    for one in files:
        labels.append(x)
        way_1=os.path.join(way,one)
        list.append(way_1)


for item in list:
    img=cv.imread(item)
    feactures.append(img)

images=np.array(feactures)
f_img=images.reshape(len(images),-1)
f_lab=np.array(labels)

model=DecisionTreeClassifier()
model.fit(f_img,f_lab)
joblib.dump(model,"rec.joblib")
    
    