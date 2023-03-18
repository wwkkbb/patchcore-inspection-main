# import matplotlib.pyplot as plt
# from sklearn import datasets
# from sklearn.decomposition import PCA, FactorAnalysis
# from sklearn.decomposition import TruncatedSVD
# from sklearn import decomposition
#
# iris = datasets.load_iris()
#
# pca = PCA()
# dt = pca.fit_transform(iris.data)
# print(pca.explained_variance_ratio_)
#
# '''
# array([  8.05814643e-01,   1.63050854e-01,   2.13486883e-02,......)
# '''
#
# fig, axes = plt.subplots(1,3)
#
# pca = decomposition.PCA(n_components = 2)
# dt = pca.fit_transform(iris.data)
# axes[0].scatter(dt[:,0], dt[:,1], c=iris.target)
#
#
# fa = FactorAnalysis(n_components=2)
# dt = fa.fit_transform(iris.data)
# axes[1].scatter(dt[:,0], dt[:,1], c=iris.target)
#
#
# svd = TruncatedSVD()
# dt = svd.fit_transform(iris.data)
# axes[2].scatter(dt[:,0], dt[:,1], c=iris.target)
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# import numpy as np
#
# X = np.array([[12, 350, 1.825, 0.102, 315, 0, 2, 4], [25, 300, 5.57, 0.45, 220, 25, 3, 2.5],  [25, 300, 5.25, 1.1, 220, 20, 4, 3]])
# Y = np.array([22, 300, 4.25, 1.86, 210, 18, 3, 2])
# # n_components指定降维后的维数
# pca = PCA(n_components=2)
# print(pca)
# # 应用于训练集数据进行PCA降维
# pca.fit(X)
# # 用X来训练PCA模型，同时返回降维后的数据
# newX = pca.fit_transform(X)
# print(newX)
# # 将降维后的数据转换成原始数据，
# pca_new = pca.transform(X)
# print(pca_new.shape)
# # 输出具有最大方差的成分
# print(pca.components_)
# # 输出所保留的n个成分各自的方差百分比
# print(pca.explained_variance_ratio_)
# # 输出所保留的n个成分各自的方差
# print(pca.explained_variance_)
# # 输出未处理的特征维数
# print(pca.n_features_)
# # 输出训练集的样本数量
# print(pca.n_samples_)
# # 输出协方差矩阵
# print(pca.noise_variance_)
# # 每个特征的奇异值
# print(pca.ingular_values_)
# # 用生成模型计算数据精度矩阵
# print(pca.get_precision())
import gc

import numpy as np
# from sklearn.metrics import confusion_matrix
# y_pred, y_true =[1,0,1,0], [0,0,1,0]
# x=confusion_matrix(y_true=y_true, y_pred=y_pred)
# print(x)
from sklearn.metrics import accuracy_score
import cv2

# y_pred是预测标签
# y_pred = cv2.imread("/home/wwkkb/MVTec/bottle/test/broken_large/000.png")
# image1 = cv2.cvtColor(y_pred, cv2.COLOR_BGR2GRAY)
#
# ret, image1 = cv2.threshold(image1, 80, 255, cv2.THRESH_BINARY)
#
# # ret,image1 = cv2.threshold(y_pred,80,255,cv2.THRESH_BINARY)
# x = accuracy_score(y_true=y_true.flatten(), y_pred=y_pred.flatten())
#
# print(1)


# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
# from sklearn import svm
# from sklearn.datasets import make_circles, make_moons, make_blobs,make_classification
# class_1 = 500 #类别1有500个样本，10：1
# class_2 = 50 #类别2只有50个
# centers = [[0.0, 0.0], [2.0, 2.0]] #设定两个类别的中心
# clusters_std = [1.5, 0.5] #设定两个类别的方差，通常来说，样本量比较大的类别会更加松散
# X, y = make_blobs(n_samples=[class_1, class_2],
#                   centers=centers,
#                   cluster_std=clusters_std,
#                   random_state=0, shuffle=False)
# #看看数据集长什么样
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap="rainbow",s=10)
# plt.show()
# #其中红色点是少数类，紫色点是多数类
# # 不设定class_weight
# clf = svm.SVC(kernel='linear', C=1.0)
# clf.fit(X, y)

# %%
import matplotlib.pylab as plt
from sklearn.metrics import roc_curve, auc,det_curve
# mask_=(mask-mask.min())/(mask.max()-mask.min())
# fpr, tpr, thresholds = roc_curve([0,1,0,1,0,1,0], [1,1,0.2,1,0.5,1,0])
#
# roc_auc = auc(fpr, tpr)
# plt.title('Receiver Operating Characteristic')
# plt.plot(fpr, tpr, '#9400D3', label=u'AUC = %0.3f' % roc_auc)
# plt.legend(loc='lower right')
# plt.plot([0, 1], [0, 1], 'r--')
# plt.xlim([-0.1, 1.1])
# plt.ylim([-0.1, 1.1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.grid(linestyle='-.')
# plt.grid(True)
# plt.show()
# print(roc_auc)
import matplotlib.cm as cm
import os
import tracemalloc

tracemalloc.start()
cur_output_dir="/home/wwkkb/result"
mask_true_grays=cv2.imread("/media/wwkkb/0CCD76FCF63D1C29/wwkkb/MVtec/bottle/test/broken_large/000.png")
# plt.figure(figsize=(30, 30), dpi=360)
# del mask_true_grays
# plt.subplot(1, 2, 1)
# x1=cv2.resize(mask_true_grays, (112,112))
# plt.imshow(x1)
# plt.subplot(1, 2, 2)
# x2=cv2.resize(mask_true_grays, (112,112))
# plt.imshow(x2)
# # plt.subplot(1, 3, 3)
# # plt.imshow(cv2.resize(mask_true_grays, (112,112)))
# os.makedirs(cur_output_dir, exist_ok=True)
# plt.savefig(os.path.join(cur_output_dir, "{:0>3}".format(str(1)) + '.png'))
# plt.colorbar()
# plt.show()
# del x1,x2,mask_true_grays
# del plt
# for i in range(100):
#     np.ones((999,999,999))
#     if i%10==0:
#         print(i)
# snapshot = tracemalloc.take_snapshot()
# top_stats = snapshot.statistics('lineno')
#
# gc.collect()
# print("[ Top 10 ]")
# for stat in top_stats[:10]:
#     print(stat)
# print(1)
import json
from decimal import getcontext, Decimal



path = r'/home/wwkkb/Downloads/project/patchcore-inspection-main/src/log/cl_1_good3_mask_unlabelwideresnet50_layer2'
name_class=os.listdir(path)
dic=dict()
with open("/home/wwkkb/aucs.txt",'w') as file:
    for i in range(len(name_class)):
        if name_class[i] in ['carpet', 'grid', 'leather', 'tile', 'wood']:
            continue
        class_name=os.path.join(path,name_class[i])
        f = open(os.path.join(class_name,"aucs.txt"), 'r')
        js = eval(f.read())
        aucs_mean=sum(js[j] for j in js)/len(js)
        aucs_mean=round(aucs_mean,4)
        dic[name_class[i]]=aucs_mean
        f.close()

    print(sum(dic[d]for d in dic)/len(dic))
    file.write(str(dic))
    file.close()

