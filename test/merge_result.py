import os
import cv2
import torch
import torchvision.transforms
import PIL.Image as Image
import numpy as np
path='/home/wwkkb/cl_wideresnet50_layer2'
path1='/home/wwkkb/MVTec'
sum=0
for i in sorted(os.listdir(path)):
    path_dir=os.path.join(path,i,"test")
    path1_dir=os.path.join(path1,i,"test")
    path_mask=os.path.join(path1,i,"ground_truth")
    for j in sorted(os.listdir(path_dir)):
        broken_dir=os.path.join(path_dir,j)
        broken1_dir=os.path.join(path1_dir,j)
        mask_dir=os.path.join(path_mask,j)
        for k in sorted(os.listdir(broken_dir)):
            # img=cv2.imread(os.path.join(broken_dir,k))
            sum+=1
            img=Image.open(os.path.join(broken_dir,k))
            _=os.path.join(mask_dir,k[:3]+'_mask'+k[-4:])
            img1=cv2.imread(os.path.join(broken1_dir,k))
            if j=='good':
                img_mask=np.zeros(img1.shape)
            else:
                img_mask=cv2.imread(_)
            cen=torchvision.transforms.CenterCrop(((min(img.size)),min(img.size)))
            img=cen(img)
            img=np.array(img)

            img = cv2.resize((cv2.cvtColor(img, cv2.COLOR_RGB2BGR)),img1.shape[:2])
            merge=np.concatenate((img,img1,img_mask),axis=1)
            print(merge.shape)
            # img=cv2.resize(img,img1.shape[:2])
            cv2.imwrite("/home/wwkkb/result/"+str(sum)+'.png',merge)
            break
