import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
#
CLASS_NAMES = [
     'bottle','cable', 'capsule', 'hazelnut',  'metal_nut', 'pill', 'screw',
    'toothbrush', 'transistor', 'zipper'
]
for class_name in CLASS_NAMES:
    img=cv2.imread(os.path.join("/home/wwkkb/MVTec",class_name,"train/good3/001.png"))
    mask=cv2.imread(os.path.join("/home/wwkkb/cl_wideresnet50_layer2",class_name,"train/good3/mask001.png"),cv2.IMREAD_GRAYSCALE)
    cv2.imshow("f_mask",mask)
    mask = cv2.GaussianBlur(mask, ksize=(11,11),sigmaX=10,sigmaY=10)
    # mask=cv2.medianBlur(mask,25)
    # print(np.unique(mask))
    temp = mask > 0
    mask = (np.where(temp, 255, 0)).astype("uint8")
    num = cv2.connectedComponents(mask,mask, 8, cv2.CV_16U)
    mask=cv2.resize(mask,(img.shape[0],img.shape[1]))
    mask=np.expand_dims(mask, axis=2)
    mask=np.uint8(np.concatenate((mask, mask,mask), axis=2)/255)
    img=mask*img
    cv2.imshow('mask',mask*255)
    cv2.imshow("x",img)
    cv2.waitKey(0)
    print(1)