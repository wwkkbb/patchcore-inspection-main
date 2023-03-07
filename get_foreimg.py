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

    for num in ['001','002','003']:
        dir_=r'train/good3'
        img=cv2.imread(os.path.join("/home/wwkkb/MVTec",class_name,dir_,num+".png"))
        mask=cv2.imread(os.path.join("/home/wwkkb/cl_wideresnet50_layer2",class_name,dir_,"mask"+num+".png"),cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (img.shape[0], img.shape[1]))

        f_mask = mask.copy()


        cv2.namedWindow("f_mask", 0)
        cv2.resizeWindow("f_mask", 600, 600)
        cv2.imshow("f_mask",mask)


        mask = cv2.GaussianBlur(mask, ksize=(11,11),sigmaX=10,sigmaY=10)
        print(np.unique(mask))
        mask=cv2.medianBlur(mask,9)
        #
        temp = mask > 0
        mask = (np.where(temp, 255, 0)).astype("uint8")
        print(np.unique(mask))
        # num = cv2.connectedComponents(mask,mask, 8, cv2.CV_16U)
        # mask=cv2.resize(mask,(img.shape[0],img.shape[1]))
        mask=np.expand_dims(mask, axis=2)
        mask=np.uint8(np.concatenate((mask, mask,mask), axis=2)/255)
        f_mask = np.expand_dims(f_mask, axis=2)
        f_mask=np.uint8(np.concatenate((f_mask, f_mask,f_mask), axis=2)/255)
        img=mask*img
        mer=np.concatenate([f_mask*255, mask*255, img], 1)
        out_path=os.path.join("/media/wwkkb/OS/wkb", class_name, dir_)
        os.makedirs( out_path,exist_ok=True)
        cv2.imwrite(os.path.join(out_path,num+"_merge.png"),mer)
        cv2.namedWindow("mask", 0)
        cv2.namedWindow("img", 0)

        cv2.resizeWindow("mask", 600, 600)
        cv2.resizeWindow("img", 600, 600)

        cv2.imshow('mask', mask * 255)
        cv2.imshow("img", img)
        # cv2.waitKey(0)
        print(1)