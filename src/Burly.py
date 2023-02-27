import os
import torch
import patchcore
import cv2 as cv
import numpy as np
from PIL import Image
from tqdm import tqdm
import patchcore.common
import patchcore.backbones
import matplotlib.pylab as plt
import scipy.ndimage as ndimage
from torchvision import transforms
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from patchcore.datasets.mvtec import _CLASSNAMES, IMAGENET_MEAN, IMAGENET_STD
from sklearn.semi_supervised import LabelPropagation
import random
from sklearn import decomposition
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.decomposition import TruncatedSVD
resize = 640
# data_root = '/home/flyinghu/Data/mvtec_anomaly_detection'
data_root = '/home/wwkkb/RegAD-main/MVTec'
backbone_name = 'wideresnet50'
layer = 'layer2'
device = 'cuda'
output_dir = f'log/cl_{backbone_name}_{layer}'
kmeans_f_num = 5000
lda_f_num = 500
# kmeans_f_num = 50000
# lda_f_num = 5000
foreground_ratio = 0.2
background_ratio = -3
lda_threshold = None  # None自动
gaussian_filter_sigma = 10
n_clusters = 4

object_classnames = ['carpet', 'grid', 'leather', 'tile', 'wood']
CLASS_NAMES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile',
    'toothbrush', 'transistor', 'wood', 'zipper'
]


class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, classname, resize=224, split: str = 'train', anomaly: str = None) -> None:
        super().__init__()
        assert classname in _CLASSNAMES
        self.root = root
        self.classname = classname
        self.resize = resize
        self.classpath = os.path.join(root, classname, 'train', 'good') if split == 'train' else os.path.join(root,
                                                                                                              classname,
                                                                                                              'test',
                                                                                                              anomaly)
        self.img_fns = [os.path.join(self.classpath, fn) for fn in os.listdir(self.classpath) if
                        os.path.isfile(os.path.join(self.classpath, fn))]
        self.transform_img = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.img_fns)

    def __getitem__(self, index):
        image_path = self.img_fns[index]
        image = Image.open(image_path).convert("RGB")
        image = self.transform_img(image)
        return image


class ForwardHook:
    def __init__(self, device='cpu'):
        self.features = []
        self.devcie = device

    def __call__(self, module, input, output):  # input:tuple:1,Tensor{1,512,80,80],output:Tensor{1,512,80,80]
        self.features.append(output.detach().to(self.devcie))
        raise patchcore.common.LastLayerToExtractReachedException()


backbone = patchcore.backbones.load(backbone_name).to(device)
backbone.eval()
network_layer = backbone.__dict__["_modules"][layer]
kmeans = KMeans(n_clusters)
lda = LinearDiscriminantAnalysis()
forward_hook = ForwardHook('cuda')
os.makedirs(output_dir, exist_ok=True)
if isinstance(network_layer, torch.nn.Sequential):
    network_layer[-1].register_forward_hook(forward_hook)
elif isinstance(network_layer, torch.nn.Module):
    network_layer.register_forward_hook(forward_hook)

for classname in CLASS_NAMES:
    # for classname in ['screw']:
    if classname in object_classnames:
        continue
    cur_classname_output_dir = os.path.join(output_dir, classname)
    os.makedirs(cur_classname_output_dir, exist_ok=True)
    print(classname)
    train_dataset = MVTecDataset(data_root, classname, resize=resize)
    for image in tqdm(train_dataset, desc='extract feature'):
        try:
            backbone(image[None].to(device))
        except patchcore.common.LastLayerToExtractReachedException:
            pass
    features = forward_hook.features
    features = torch.cat(features, 0)  # b x 512 x h x w
    image_features = features.permute(0, 2, 3, 1).cpu().numpy()  # b x h x w x 512

    # n_component_pca=512
    # iris = image_features.reshape(-1,image_features.shape[3])
    # b,h,w,c=image_features.shape
    # # pca = PCA()
    # # dt = pca.fit_transform(iris)
    # # x=sum(pca.explained_variance_ratio_[:256])
    # pca = decomposition.PCA(n_components=n_component_pca)
    # iris= pca.fit_transform(iris)
    #
    # # x=pca.transform(iris[:10])
    # image_features=iris.reshape(b,h,w,n_component_pca)
    # features=(torch.tensor(image_features).permute(0,3,1,2))
    # # features=features.numpy()
    # x=[]
    # for i in features:
    #     x.append(i.unsqueeze(0))
    # forward_hook.features=x


    print('kmeans')
    kmeans.fit(image_features.reshape(-1, features.shape[1])[
                   np.random.permutation(image_features.reshape(-1, features.shape[1]).shape[0])[:kmeans_f_num]])
    # labels = kmeans.labels_
    print('kmeans predict')
    labels = kmeans.predict(image_features.reshape(-1, features.shape[1]))
    labels_imgs = labels.reshape(len(forward_hook.features), forward_hook.features[0].shape[2],
                                 forward_hook.features[0].shape[3])
    if background_ratio < 0:
        background_ratio = -background_ratio / labels_imgs.shape[1]
    # 以ratio作为边界框，选取边界框到边界的值统计hist
    background_mask = np.zeros(
        (len(forward_hook.features), forward_hook.features[0].shape[2], forward_hook.features[0].shape[3]), dtype=bool)
    background_mask[:, :int(background_ratio * labels_imgs.shape[1]), :] = True
    background_mask[:, -int(background_ratio * labels_imgs.shape[1]):, :] = True
    background_mask[:, int(background_ratio * labels_imgs.shape[1]):-int(background_ratio * labels_imgs.shape[1]),
    :int(background_ratio * labels_imgs.shape[2])] = True
    background_mask[:, int(background_ratio * labels_imgs.shape[1]):-int(background_ratio * labels_imgs.shape[1]),
    -int(background_ratio * labels_imgs.shape[2]):] = True
    bidx, hidx, widx = np.where(background_mask)
    background_features = image_features[bidx, hidx, widx, :]
    background_labels = labels_imgs[bidx, hidx, widx]
    one_hot = np.zeros((background_labels.shape[0], kmeans.n_clusters))
    one_hot[np.arange(one_hot.shape[0]), background_labels] = 1
    hist = one_hot.sum(0)
    # background_label_u = np.arange(kmeans.n_clusters)[hist > one_hot.shape[0] / kmeans.n_clusters]
    background_label_u = [hist.argmax()]
    background_p_mask = (np.stack([background_labels == l for l in background_label_u], 1).sum(1) > 0)
    background_features = background_features[background_p_mask]
    background_label = np.zeros((background_features.shape[0]), dtype=int)

    # 前景
    foreground_mask = np.zeros(
        (len(forward_hook.features), forward_hook.features[0].shape[2], forward_hook.features[0].shape[3]), dtype=bool)
    foreground_mask[:, int(labels_imgs.shape[1] / 2 - labels_imgs.shape[1] * foreground_ratio):int(
        labels_imgs.shape[1] / 2 + labels_imgs.shape[1] * foreground_ratio),
    int(labels_imgs.shape[2] / 2 - labels_imgs.shape[2] * foreground_ratio):int(
        labels_imgs.shape[2] / 2 + labels_imgs.shape[2] * foreground_ratio)] = True

    bidx, hidx, widx = np.where(foreground_mask)
    foreground_features = image_features[bidx, hidx, widx, :]
    foreground_labels = labels_imgs[bidx, hidx, widx]
    foreground_p_mask = (
            np.stack([foreground_labels != l for l in background_label_u], 1).sum(1) >= len(background_label_u))
    foreground_features = foreground_features[foreground_p_mask]
    foreground_label = np.ones((foreground_features.shape[0]), dtype=int)
    background_idx = np.random.permutation(len(background_features))[:lda_f_num]
    foreground_idx = np.random.permutation(len(foreground_features))[:lda_f_num]
    background_features = background_features[background_idx]
    foreground_features = foreground_features[foreground_idx]
    background_label = background_label[background_idx]
    foreground_label = foreground_label[foreground_idx]

    # data_all = np.vstack((background_features, foreground_features))
    # label_all = np.hstack((background_label, foreground_label))
    # rng = np.random.RandomState(0)
    # indices = np.arange(len(data_all))
    # rng.shuffle(indices)
    # rng=indices
    # data_all = data_all[rng]
    # label_all=label_all[rng]
    # rate=2
    # unlabeled_data_true=label_all[int(len(data_all)*rate/10):]
    # unlabeled_data_true=unlabeled_data_true.copy()
    # label_all[int(len(data_all)*rate/10):]=-1
    # # labeled_label = label_all[rng[:int(len(data_all) / 10)]]
    # # unlabeled_data = data_all[rng[int(len(data_all) / 10):]]
    # # unlabeled_label = label_all[rng[int(len(data_all) / 10):]]
    # clf = LabelPropagation(max_iter=100, kernel='rbf', gamma=10)
    # clf.fit(data_all, label_all)
    #
    # print('Accuracy:%f' % clf.score(data_all[int(len(data_all)*rate/10):], unlabeled_data_true))


    def test_LabelPropagation(*data):
        X, y, unlabeled_data = data
        unlabeled_data=unlabeled_data[0:1000]
        unlabeled_data=unlabeled_data.copy()
        train_unlabel=-np.ones(len(unlabeled_data))
        train_all=np.vstack((X,unlabeled_data))
        label_all=np.hstack((y,train_unlabel))

        clf = LabelPropagation(max_iter=1000, kernel='rbf', gamma=5)
        clf.fit(train_all, label_all)
        return train_all,clf.predict(train_all)

        # 获取预测准确率
        # print('Accuracy:%f' % clf.score(X[unlabeled_indices], true_labels))
    labeled_data = np.vstack((background_features, foreground_features))
    labeled_label = np.hstack((background_label, foreground_label))
    train,label=test_LabelPropagation(labeled_data,labeled_label,image_features.reshape(-1, features.shape[1]))


    lda.fit(train,label)
    cur_output_dir = os.path.join(cur_classname_output_dir, 'train', 'good')
    os.makedirs(cur_output_dir, exist_ok=True)
    for i in tqdm(range(len(train_dataset)), desc='train result', leave=False):
        image_path = train_dataset.img_fns[i]
        image_name = os.path.basename(image_path)
        image = Image.open(image_path).convert("RGB")
        image = train_dataset.transform_img(image)
        forward_hook.features = []
        try:
            backbone(image[None].to(device))
        except patchcore.common.LastLayerToExtractReachedException:
            pass
        features = torch.cat(forward_hook.features, 0)  # b x 512 x h x w


        # def PCA_w(features,pca):
        #     b, c, h, w = features.shape
        #     features = (features.reshape(-1, features.shape[1])).cpu().numpy()
        #     features = torch.tensor(pca.transform(features))
        #     features = features.cuda()
        #     return features,b,h,w
        # features,b,h,w=PCA_w(features,pca)
        # features=features.reshape(b,n_component_pca,h,w)


        features = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1]).cpu().numpy()
        lda_predict = lda.transform(features)
        lda_predict = lda_predict.reshape(forward_hook.features[0].shape[2], forward_hook.features[0].shape[3])
        lda_predict = cv.resize(lda_predict, (image.shape[2], image.shape[1]))
        lda_predict = ndimage.gaussian_filter(lda_predict, sigma=gaussian_filter_sigma)
        # 将结果二值化然后求最大连通域的最小外接矩形。
        if lda_threshold is None:
            lda_mask = lda.predict(features).reshape(forward_hook.features[0].shape[2],
                                                     forward_hook.features[0].shape[3])
            lda_mask = cv.resize(lda_mask, (image.shape[2], image.shape[1]), interpolation=cv.INTER_NEAREST).astype(
                np.uint8) * 255
        else:
            lda_mask = (lda_predict > lda_threshold).astype(np.uint8) * 255
        # cv.imshow('graycsale image', lda_mask)
        cv.imwrite(os.path.join(cur_output_dir, 'mask'+image_name),lda_mask)

        # waitKey() waits for a key press to close the window and 0 specifies indefinite loop
        cv.waitKey(0)


        cn, cc_labels, cc_stats, _ = cv.connectedComponentsWithStats(lda_mask, connectivity=8)
        mask = np.zeros_like(lda_predict, dtype=np.uint8)
        if cn > 1:
            max_cc_labels = np.argmax(np.array([i[-1] for i in cc_stats[1:]])) + 1
            y, x = np.where(cc_labels == max_cc_labels)
            rect = cv.minAreaRect(np.stack([x, y], axis=1))
            box = cv.boxPoints(rect)
            mask = cv.fillPoly(mask, box[None].astype(int), 255)
        plt.figure(figsize=(15, 15), dpi=180)
        plt.subplot(1, 3, 1)
        plt.imshow(image.permute(1, 2, 0) * torch.tensor(IMAGENET_STD) + torch.tensor(IMAGENET_MEAN))
        plt.subplot(1, 3, 2)
        plt.imshow(lda_predict, plt.cm.hot)
        plt.subplot(1, 3, 3)
        plt.imshow(mask, 'gray')
        plt.savefig(os.path.join(cur_output_dir, image_name))
        plt.close()
        # cv.imwrite(os.path.join(cur_output_dir, image_name), mask)

    # 测试数据集
    for an in tqdm(os.listdir(os.path.join(train_dataset.root, train_dataset.classname, 'test')), 'test result'):
        test_dataset = MVTecDataset(train_dataset.root, train_dataset.classname, train_dataset.resize, split='test',
                                    anomaly=an)
        cur_output_dir = os.path.join(cur_classname_output_dir, 'test', f'{an}')
        os.makedirs(cur_output_dir, exist_ok=True)
        for i in tqdm(range(len(test_dataset)), desc=f'test {an} result', leave=False):
            image_path = test_dataset.img_fns[i]
            image_name = os.path.basename(image_path)
            image = Image.open(image_path).convert("RGB")
            image = test_dataset.transform_img(image)
            forward_hook.features = []
            try:
                backbone(image[None].to(device))
            except patchcore.common.LastLayerToExtractReachedException:
                pass
            features = torch.cat(forward_hook.features, 0)  # b x 512 x h x w


            # def PCA_w(features, pca):
            #     b, c, h, w = features.shape
            #     features = (features.reshape(-1, features.shape[1])).cpu().numpy()
            #     features = torch.tensor(pca.transform(features))
            #     features = features.cuda()
            #     return features, b, h, w
            # features, b, h, w = PCA_w(features, pca)
            # features = features.reshape(b, n_component_pca, h, w)


            features = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1]).cpu().numpy()
            lda_predict = lda.transform(features)
            lda_predict = lda_predict.reshape(forward_hook.features[0].shape[2], forward_hook.features[0].shape[3])
            lda_predict = cv.resize(lda_predict, (image.shape[2], image.shape[1]))
            lda_predict = ndimage.gaussian_filter(lda_predict, sigma=gaussian_filter_sigma)

            # 将结果二值化然后求最大连通域的最小外接矩形。
            if lda_threshold is None:
                lda_mask = lda.predict(features).reshape(forward_hook.features[0].shape[2],
                                                         forward_hook.features[0].shape[3])
                lda_mask = cv.resize(lda_mask, (image.shape[2], image.shape[1]), interpolation=cv.INTER_NEAREST).astype(
                    np.uint8) * 255
            else:
                lda_mask = (lda_predict > lda_threshold).astype(np.uint8) * 255
            cn, cc_labels, cc_stats, _ = cv.connectedComponentsWithStats(lda_mask, connectivity=8)
            mask = np.zeros_like(lda_predict, dtype=np.uint8)
            if cn > 1:
                max_cc_labels = np.argmax(np.array([i[-1] for i in cc_stats[1:]])) + 1
                y, x = np.where(cc_labels == max_cc_labels)
                rect = cv.minAreaRect(np.stack([x, y], axis=1))
                box = cv.boxPoints(rect)
                mask = cv.fillPoly(mask, box[None].astype(int), 255)
            plt.figure(figsize=(15, 15), dpi=180)
            plt.subplot(1, 3, 1)
            plt.imshow(image.permute(1, 2, 0) * torch.tensor(IMAGENET_STD) + torch.tensor(IMAGENET_MEAN))
            plt.subplot(1, 3, 2)
            plt.imshow(lda_predict, plt.cm.hot)
            plt.subplot(1, 3, 3)
            plt.imshow(mask, 'gray')
            plt.savefig(os.path.join(cur_output_dir, image_name))
            plt.close()
            # cv.imwrite(os.path.join(cur_output_dir, image_name), mask)
