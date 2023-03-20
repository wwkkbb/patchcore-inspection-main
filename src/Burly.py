import os
import gc
import heapq
import cv2
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
from patchcore import sampler
import patchcore
import patchcore
from patchcore import patchcore_Burly as patchcore_model
from torchvision import models
import random
from sklearn import decomposition
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.decomposition import TruncatedSVD
import matplotlib.pylab as plt
import matplotlib.cm as cm
import dill as pickle
resize = 640
# data_root = '/home/flyinghu/Data/mvtec_anomaly_detection'

backbone_name = 'wideresnet50'
layer = 'layer2'
device = 'cuda'
#train
data_root = '/media/wwkkb/0CCD76FCF63D1C29/wwkkb/MVTec'
kmeans_f_rate = 0.01
rate_unlabel = 0.02

lda_f_num = 500
output_dir = f'log/cl_1_good320_mask_unlabel{backbone_name}_{layer}'
#debug
# data_root = '/media/wwkkb/0CCD76FCF63D1C29/wwkkb/MVtec'
# kmeans_f_rate = 0.01
# rate_unlabel = 0.001
# lda_f_num = 200
# output_dir = f'log/cl_1_good1_mask_unlabel{backbone_name}_{layer}'

# kmeans_f_num = 50000
# lda_f_num = 5000
foreground_ratio = 0.2
background_ratio = -3
lda_threshold = None  # None自动
gaussian_filter_sigma = 10
n_clusters = 4

object_classnames = ['carpet', 'grid', 'leather', 'tile', 'wood','bottle','cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',]
CLASS_NAMES = [
      'metal_nut', 'pill', 'screw', 'tile',
    'toothbrush', 'transistor', 'wood', 'zipper'
]


class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, classname, resize=224, split: str = 'train', anomaly: str = None) -> None:
        super().__init__()
        assert classname in _CLASSNAMES
        self.root = root
        self.classname = classname
        self.resize = resize
        self.classpath = os.path.join(root, classname, 'train', 'good3') if split == 'train' else os.path.join(root,
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
                   np.random.permutation(image_features.reshape(-1, features.shape[1]).shape[0])[
                   :(kmeans_f_rate * np.sum(image_features.shape)).astype(int)]])
    # labels = kmeans.labels_
    print('kmeans predict')

    #Save the kmeans model
    pickle.dump(kmeans, open(os.path.join(output_dir ,classname,"kmeans.dat"), "wb"))
    # kmeans = pickle.load(open(os.path.join(output_dir ,classname, "kmeans.dat"), "rb"))
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
        X, y, unlabeled_data_all = data
        sampling_percentage = 0.1
        feature_dimension = 512
        # model = sampler.GreedyCoresetSampler(
        #     percentage=sampling_percentage,
        #     device=torch.device("cpu"),
        #     dimension_to_project_features_to=feature_dimension,
        # )
        # unlabeled_datas = None
        # num = 20
        # for sa in range(num):
        #     unlabeled_data = torch.from_numpy(unlabeled_data_all[int(len(unlabeled_data_all) / num) * sa:int(
        #         len(unlabeled_data_all) / num) * (sa + 1)])
        #     unlabeled_data = model.run(unlabeled_data)
        #     if unlabeled_datas == None:
        #         unlabeled_datas = unlabeled_data
        #     else:
        #         unlabeled_datas = torch.cat((unlabeled_datas, unlabeled_data), dim=0)
        # unlabeled_data=model.run(unlabeled_datas)
        idx = np.random.permutation(len(unlabeled_data_all))[:int(rate_unlabel * len(unlabeled_data_all))]
        unlabeled_data = (unlabeled_data_all[idx])
        unlabeled_data = unlabeled_data.copy()
        train_unlabel = -np.ones(len(unlabeled_data))
        train_all = np.vstack((X, unlabeled_data))
        label_all = np.hstack((y, train_unlabel))

        clf = LabelPropagation(max_iter=1000, kernel='rbf', gamma=5)
        clf.fit(train_all, label_all)
        # train_all = train_all[-len(unlabeled_data):]
        sum = 0
        # labels=[]
        div = 20
        devide = int(len(unlabeled_data_all) / div)
        for i in range(div):
            pre = unlabeled_data_all[i * devide:(i + 1) * devide]
            label = clf.predict(pre)
            index_ = np.where(label == 1)
            x = pre[index_]
            unlabeled_data_all[sum:sum + len(index_[0])] = x
            sum += len(index_[0])
        # return unlabeled_data_all[:sum], clf #train
        return unlabeled_data_all[:sum], clf#debug

        # 获取预测准确率
        # print('Accuracy:%f' % clf.score(X[unlabeled_indices], true_labels))


    labeled_data = np.vstack((background_features, foreground_features))
    labeled_label = np.hstack((background_label, foreground_label))
    trains, clf = test_LabelPropagation(labeled_data, labeled_label,
                                        image_features.reshape(-1, features.shape[1]))
    # 保存模型,我们想要导入的是模型本身，所以用“wb”方式写入，即是二进制方式,DT是模型名字
    # Save the kmeans model
    pickle.dump(clf, open(os.path.join(output_dir ,classname,"dtr.dat"), "wb"))
    # clf = pickle.load(open(os.path.join(output_dir ,classname,"dtr.dat"), "rb"))
    image_dimension = 112


    def _standard_patchcore(image_dimension):
        patchcore_instance = patchcore_model.PatchCore(torch.device("cpu"))
        backbone = models.wide_resnet50_2(pretrained=False)
        backbone.name, backbone.seed = "wideresnet50", 0
        patchcore_instance.load(
            backbone=backbone,
            layers_to_extract_from=["layer2", "layer3"],
            device=torch.device("cpu"),
            input_shape=[3, image_dimension, image_dimension],
            pretrain_embed_dimension=1024,
            target_embed_dimension=1024,
            patchsize=3,
            patchstride=1,
            spade_nn=2,
            featuresampler=patchcore.sampler.ApproximateGreedyCoresetSampler(percentage=0.1,device="cpu")
        )
        return patchcore_instance


    model = _standard_patchcore(image_dimension)
    model.anomaly_scorer = patchcore.common.NearestNeighbourScorer(
            n_nearest_neighbours=1, nn_method=patchcore.common.ApproximateFaissNN(False, 4)
        )
    model.fit(trains)
    # Save the patchcore model
    pickle.dump(model, open(os.path.join(output_dir ,classname,"patchcore.dat"), "wb"))
    # test
    x = os.listdir(os.path.join(train_dataset.root, train_dataset.classname, 'test'))
    for i in range(len(x)):
        if x[i] == 'good':
            x[i] = x[0]
            x[0] = 'good'
    dic=dict()
    for an in tqdm(x, 'test result'):
        test_dataset = MVTecDataset(train_dataset.root, train_dataset.classname, train_dataset.resize, split='test',
                                    anomaly=an)

        cur_output_dir = os.path.join(cur_classname_output_dir, 'test', f'{an}')
        os.makedirs(cur_output_dir, exist_ok=True)
        max_s = []
        min_s = []
        acc_s = []
        forward_hook.features = []
        mask_true_grays = []
        shape_end = (112, 112)
        for i in tqdm(range(len(test_dataset)), desc=f'test {an} result'):
            # for i in tqdm(range(len(test_dataset)), desc=f'test {an} result', leave=False):
            image_path = test_dataset.img_fns[i]
            image_name = os.path.basename(image_path)
            image = Image.open(image_path).convert("RGB")
            image_test = image
            image_test = np.array(image_test).astype(np.uint8)
            image = test_dataset.transform_img(image)
            try:
                backbone(image[None].to(device))
            except patchcore.common.LastLayerToExtractReachedException:
                pass

            if f'{an}' == 'good':
                # mask_true = np.zeros((mask.shape[0], mask.shape[1], 3))
                mask_true_gray = np.zeros((shape_end[0], shape_end[1]))
            else:
                # mask_true = cv2.imread(cur_classname_mask)
                # mask_true = cv.cvtColor(mask_true, cv.COLOR_BGR2RGB)
                cur_classname_mask = os.path.join(train_dataset.root, train_dataset.classname, 'ground_truth', f'{an}',
                                                  "{:0>3}".format(str(i)) + '_mask.png')
                mask_true_gray = cv2.resize(cv2.imread(cur_classname_mask, 0), shape_end)
            mask_true_grays.append(mask_true_gray)
        del mask_true_gray
        features = torch.cat(forward_hook.features, 0)  # b x 512 x h x w
        shape_ = (features.shape[0], features.shape[2], features.shape[3])
        features = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1]).cpu().numpy()

        labels = clf.predict(features)


        def get_accuracy(scores, true_mask):
            from heapq import heapify as heapify
            from sklearn.metrics import accuracy_score
            if len(true_mask.shape) == 3:
                true_mask = cv2.cvtColor(true_mask, cv2.COLOR_RGB2GRAY)
            max_ = scores.max()
            min_ = scores.min()
            pre = cv2.resize(((max_ - scores) / (max_ - min_)) * 255, true_mask.shape).astype('uint8')
            # if ret=='good':
            #     scores_1=list(scores.flatten())
            #     heapify(scores_1)
            #     for _ in range(len(scores_1)/10):
            #
            ret, pre = cv2.threshold(pre, 250, 255, cv2.THRESH_BINARY)
            return accuracy_score(pre.flatten(), true_mask.flatten()), max_, min_

        with torch.no_grad():
            model = pickle.load(open(os.path.join(output_dir, classname, "patchcore.dat"), "rb"))
            scores, masks = model.predict((torch.from_numpy(features), shape_))


            # h=pow(len(label),0.5)
            def get_auc(mask_true_gray, mask,cur_output_dir):
                from sklearn.metrics import roc_curve, auc
                mask_true_gray=np.array(mask_true_gray)
                mask_true_gray[:,0,0]=1
                mask_true_gray = (np.concatenate(mask_true_gray)).flatten()
                mask = (np.concatenate(mask)).flatten()
                mask_ = (mask - mask.min()) / (mask.max() - mask.min())
                mask_true_gray = mask_true_gray.astype(int)
                mask_true_gray[0] = 1
                fpr, tpr, thresholds = roc_curve(mask_true_gray.flatten(), mask_.flatten(), pos_label=1)

                roc_auc = auc(fpr, tpr)
                plt.title('Receiver Operating Characteristic')
                plt.plot(fpr, tpr, '#9400D3', label=u'AUC = %0.3f' % roc_auc)
                plt.legend(loc='lower right')
                plt.plot([0, 1], [0, 1], 'r--')
                plt.xlim([-0.1, 1.1])
                plt.ylim([-0.1, 1.1])
                plt.ylabel('True Positive Rate')
                plt.xlabel('False Positive Rate')
                plt.grid(linestyle='-.')
                plt.grid(True)
                plt.savefig(os.path.join(cur_output_dir, 'auc.png'))
                # plt.show()
                print(roc_auc)
                return roc_auc

            dic[an]=get_auc(mask_true_grays, masks,cur_output_dir)

            _ = np.zeros((shape_[0], 112, 112))
            labels = labels.reshape((shape_[0], shape_[1], shape_[2]))
            for i in range(len(labels)):
                _[i] = masks[i] * cv2.resize(labels[i], (112, 112))
            masks = _
            for i in range(len(labels)):
                # mask = np.where(mask < min_, min_, mask)

                # acc, max_, min_ = get_accuracy(masks[i], mask_true_grays[i])
                # min_s.append(min_)
                # max_s.append(max_)
                # acc_s.append(acc)
                # # plt.figure(figsize=(30, 30), dpi=360)
                # plt.subplot(1, 3, 1)
                # plt.imshow(cv2.resize(mask_true_grays[i], masks[i].shape),cmap=plt.get_cmap('gray'))
                # plt.subplot(1, 3, 2)
                # plt.imshow(cv2.resize(image_test, masks[i].shape))
                # plt.subplot(1, 3, 3)
                # plt.imshow(masks[i], cmap=cm.hot)
                # os.makedirs(cur_output_dir, exist_ok=True)
                # plt.savefig(os.path.join(cur_output_dir, "{:0>3}".format(str(i)) + '.png'))
                # plt.colorbar()
                # # plt.show()
                acc, max_, min_ = get_accuracy(masks[i], mask_true_grays[i])
                min_s.append(min_)
                max_s.append(max_)
                acc_s.append(acc)
                # plt.figure(figsize=(30, 30), dpi=360)
                plt.subplot(1, 3, 1)
                x1 = cv2.resize(mask_true_grays[i], masks[i].shape)
                plt.imshow(x1, cmap=plt.get_cmap('gray'))
                plt.subplot(1, 3, 2)
                x2 = cv2.resize(image_test, masks[i].shape)
                plt.imshow(x2)
                plt.subplot(1, 3, 3)
                plt.imshow(masks[i], cmap=cm.hot)
                os.makedirs(cur_output_dir, exist_ok=True)

                plt.savefig(os.path.join(cur_output_dir, "{:0>3}".format(str(i)) + '.png'))
                plt.colorbar()
                # plt.show()
                plt.close()
    with open(os.path.join(cur_classname_output_dir, 'aucs.txt'), 'w') as file:
        file.write(str(dic))
    del model
    del features
    del clf,labels_imgs,labels
    gc.collect()
    torch.cuda.empty_cache()




















    # model = pickle.load(open(os.path.join(output_dir,classname ,"patchcore.dat"), "rb"))
    # test
    # x = os.listdir(os.path.join(train_dataset.root, train_dataset.classname, 'test'))
    # for i in range(len(x)):
    #     if x[i] == 'good':
    #         x[i] = x[0]
    #         x[0] = 'good'
    # for an in tqdm(x, 'test result'):
    #     test_dataset = MVTecDataset(train_dataset.root, train_dataset.classname, train_dataset.resize, split='test',
    #                                 anomaly=an)
    #
    #     cur_output_dir = os.path.join(cur_classname_output_dir, 'test', f'{an}')
    #     os.makedirs(cur_output_dir, exist_ok=True)
    #     max_s = []
    #     min_s = []
    #     acc_s = []
    #     for i in tqdm(range(len(test_dataset)), desc=f'test {an} result', leave=False):
    #         image_path = test_dataset.img_fns[i]
    #         image_name = os.path.basename(image_path)
    #         image = Image.open(image_path).convert("RGB")
    #         image_test = image
    #         image_test = np.array(image_test).astype(np.uint8)
    #         image = test_dataset.transform_img(image)
    #         forward_hook.features = []
    #         try:
    #             backbone(image[None].to(device))
    #         except patchcore.common.LastLayerToExtractReachedException:
    #             pass
    #         features = torch.cat(forward_hook.features, 0)  # b x 512 x h x w
    #         features = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1]).cpu().numpy()
    #         label = clf.predict(features)
    #
    #
    #         # image_features = features.permute(0, 2, 3, 1).cpu().numpy()  # b x h x w x 512
    #         #
    #         # print('kmeans')
    #         # kmeans.fit(image_features.reshape(-1, features.shape[1])[
    #         #                np.random.permutation(image_features.reshape(-1, features.shape[1]).shape[0])[
    #         #                :kmeans_f_num]])
    #         # # labels = kmeans.labels_
    #         # print('kmeans predict')
    #         # labels = kmeans.predict(image_features.reshape(-1, features.shape[1]))
    #         # labels_imgs = labels.reshape(len(forward_hook.features), forward_hook.features[0].shape[2],
    #         #                              forward_hook.features[0].shape[3])
    #         # if background_ratio < 0:
    #         #     background_ratio = -background_ratio / labels_imgs.shape[1]
    #         # # 以ratio作为边界框，选取边界框到边界的值统计hist
    #         # background_mask = np.zeros(
    #         #     (len(forward_hook.features), forward_hook.features[0].shape[2], forward_hook.features[0].shape[3]),
    #         #     dtype=bool)
    #         # background_mask[:, :int(background_ratio * labels_imgs.shape[1]), :] = True
    #         # background_mask[:, -int(background_ratio * labels_imgs.shape[1]):, :] = True
    #         # background_mask[:,
    #         # int(background_ratio * labels_imgs.shape[1]):-int(background_ratio * labels_imgs.shape[1]),
    #         # :int(background_ratio * labels_imgs.shape[2])] = True
    #         # background_mask[:,
    #         # int(background_ratio * labels_imgs.shape[1]):-int(background_ratio * labels_imgs.shape[1]),
    #         # -int(background_ratio * labels_imgs.shape[2]):] = True
    #         # bidx, hidx, widx = np.where(background_mask)
    #         # background_features = image_features[bidx, hidx, widx, :]
    #         # background_labels = labels_imgs[bidx, hidx, widx]
    #         # one_hot = np.zeros((background_labels.shape[0], kmeans.n_clusters))
    #         # one_hot[np.arange(one_hot.shape[0]), background_labels] = 1
    #         # hist = one_hot.sum(0)
    #         # # background_label_u = np.arange(kmeans.n_clusters)[hist > one_hot.shape[0] / kmeans.n_clusters]
    #         # background_label_u = [hist.argmax()]
    #         # background_p_mask = (np.stack([background_labels == l for l in background_label_u], 1).sum(1) > 0)
    #         # background_features = background_features[background_p_mask]
    #         # background_label = np.zeros((background_features.shape[0]), dtype=int)
    #         #
    #         # # 前景
    #         # foreground_mask = np.zeros(
    #         #     (len(forward_hook.features), forward_hook.features[0].shape[2], forward_hook.features[0].shape[3]),
    #         #     dtype=bool)
    #         # foreground_mask[:, int(labels_imgs.shape[1] / 2 - labels_imgs.shape[1] * foreground_ratio):int(
    #         #     labels_imgs.shape[1] / 2 + labels_imgs.shape[1] * foreground_ratio),
    #         # int(labels_imgs.shape[2] / 2 - labels_imgs.shape[2] * foreground_ratio):int(
    #         #     labels_imgs.shape[2] / 2 + labels_imgs.shape[2] * foreground_ratio)] = True
    #         #
    #         # bidx, hidx, widx = np.where(foreground_mask)
    #         # foreground_features = image_features[bidx, hidx, widx, :]
    #         # foreground_labels = labels_imgs[bidx, hidx, widx]
    #         # foreground_p_mask = (
    #         #         np.stack([foreground_labels != l for l in background_label_u], 1).sum(1) >= len(background_label_u))
    #         # foreground_features = foreground_features[foreground_p_mask]
    #         # foreground_label = np.ones((foreground_features.shape[0]), dtype=int)
    #         # background_idx = np.random.permutation(len(background_features))[:lda_f_num]
    #         # foreground_idx = np.random.permutation(len(foreground_features))[:lda_f_num]
    #         # background_features = background_features[background_idx]
    #         # foreground_features = foreground_features[foreground_idx]
    #         # background_label = background_label[background_idx]
    #         # foreground_label = foreground_label[foreground_idx]
    #         #
    #         # def test_LabelPropagation(*data):
    #         #     X, y, unlabeled_data_all = data
    #         #     # sampling_percentage = 0.05
    #         #     # feature_dimension = 512
    #         #     # model = sampler.GreedyCoresetSampler(
    #         #     #     percentage=sampling_percentage,
    #         #     #     device=torch.device("cpu"),
    #         #     #     dimension_to_project_features_to=feature_dimension,
    #         #     # )
    #         #     # unlabeled_datas = None
    #         #     # num = 20
    #         #     # for sa in range(num):
    #         #     #     unlabeled_data = torch.from_numpy(unlabeled_data_all[int(len(unlabeled_data_all) / num) * sa:int(
    #         #     #         len(unlabeled_data_all) / num) * (sa + 1)])
    #         #     #     unlabeled_data = model.run(unlabeled_data)
    #         #     #     if unlabeled_datas == None:
    #         #     #         unlabeled_datas = unlabeled_data
    #         #     #     else:
    #         #     #         unlabeled_datas = torch.cat((unlabeled_datas, unlabeled_data), dim=0)
    #         #     # unlabeled_data = model.run(unlabeled_datas)
    #         #     # unlabeled_data = unlabeled_data.numpy()
    #         #     # unlabeled_data = unlabeled_data.copy()
    #         #     train_unlabel = -np.ones(len(unlabeled_data_all))
    #         #     train_all = np.vstack((X, unlabeled_data_all))
    #         #     label_all = np.hstack((y, train_unlabel))
    #         #
    #         #     clf = LabelPropagation(max_iter=500, kernel='rbf', gamma=5)
    #         #     clf.fit(train_all, label_all)
    #         #     return train_all[-len(unlabeled_data_all):], clf.predict(train_all)[-len(unlabeled_data_all):]
    #         #
    #         #     # 获取预测准确率
    #         #     # print('Accuracy:%f' % clf.score(X[unlabeled_indices], true_labels))
    #         #
    #         #
    #         # labeled_data = np.vstack((background_features, foreground_features))
    #         # labeled_label = np.hstack((background_label, foreground_label))
    #         # train, label = test_LabelPropagation(labeled_data, labeled_label,
    #         #                                      image_features.reshape(-1, features.shape[1]))
    #         # features=((train.T)*label).T
    #         #
    #         # print(features.shape)
    #         def get_accuracy(scores, true_mask):
    #             from heapq import heapify as heapify
    #             from sklearn.metrics import accuracy_score
    #             if len(true_mask.shape) == 3:
    #                 true_mask = cv2.cvtColor(true_mask, cv2.COLOR_RGB2GRAY)
    #             max_ = scores.max()
    #             min_ = scores.min()
    #             pre = cv2.resize(((max_ - scores) / (max_ - min_)) * 255, true_mask.shape).astype('uint8')
    #             # if ret=='good':
    #             #     scores_1=list(scores.flatten())
    #             #     heapify(scores_1)
    #             #     for _ in range(len(scores_1)/10):
    #             #
    #             ret, pre = cv2.threshold(pre, 250, 255, cv2.THRESH_BINARY)
    #             return accuracy_score(pre.flatten(), true_mask.flatten()), max_, min_
    #
    #
    #         scores, masks = model.predict(torch.from_numpy(features))
    #         # h=pow(len(label),0.5)
    #         label = label.reshape((80, 80))
    #         label = cv2.resize(label, (112, 112))
    #
    #         mask = masks[0] * label
    #         # mask = np.where(mask < min_, min_, mask)
    #         cur_classname_mask = os.path.join(train_dataset.root, train_dataset.classname, 'ground_truth', f'{an}',
    #                                           image_name[:3] + '_mask.png')
    #
    #         mask = cv2.resize(mask, (224, 224))
    #         if f'{an}' == 'good':
    #             mask_true = np.zeros((mask.shape[0], mask.shape[1], 3))
    #             mask_true_gray = np.zeros((mask.shape[0], mask.shape[1]))
    #         else:
    #             mask_true = cv2.imread(cur_classname_mask)
    #             mask_true = cv.cvtColor(mask_true, cv.COLOR_BGR2RGB)
    #             mask_true_gray = cv2.imread(cur_classname_mask, 0)
    #         acc, max_, min_ = get_accuracy(masks[0], mask_true_gray)
    #         from sklearn.metrics import roc_curve, auc
    #         mask_=(mask-mask.min())/(mask.max()-mask.min())
    #         mask_true_gray=mask_true_gray.astype(int)
    #         mask_true_gray[0][0]=1
    #         fpr, tpr, thresholds = roc_curve(mask_true_gray.flatten(), mask_.flatten(),pos_label=1)
    #
    #         roc_auc = auc(fpr, tpr)
    #         plt.title('Receiver Operating Characteristic')
    #         plt.plot(fpr, tpr, '#9400D3', label=u'AUC = %0.3f' % roc_auc)
    #         plt.legend(loc='lower right')
    #         plt.plot([0, 1], [0, 1], 'r--')
    #         plt.xlim([-0.1, 1.1])
    #         plt.ylim([-0.1, 1.1])
    #         plt.ylabel('True Positive Rate')
    #         plt.xlabel('False Positive Rate')
    #         plt.grid(linestyle='-.')
    #         plt.grid(True)
    #         plt.show()
    #         print(roc_auc)
    #
    #
    #         min_s.append(min_)
    #         max_s.append(max_)
    #         acc_s.append(acc)
    #         mask[0][0] = 11.78
    #         plt.figure(figsize=(30, 30), dpi=360)
    #         plt.subplot(1, 3, 1)
    #         plt.imshow(cv2.resize(mask_true, mask.shape))
    #         plt.subplot(1, 3, 2)
    #         plt.imshow(cv2.resize(image_test, mask.shape))
    #         plt.subplot(1, 3, 3)
    #         plt.imshow(mask, cmap=cm.hot)
    #         os.makedirs(cur_output_dir, exist_ok=True)
    #         plt.savefig(os.path.join(cur_output_dir, image_name))
    #         plt.colorbar()
    #         plt.show()
    #     with open(os.path.join(cur_output_dir, 'max_min_acc.txt'), 'w') as file:
    #         file.write(str([min_s, max_s, acc_s]))



    # lda.fit(train,label)
    # cur_output_dir = os.path.join(cur_classname_output_dir, 'train', 'good1')
    # os.makedirs(cur_output_dir, exist_ok=True)
    # for i in tqdm(range(len(train_dataset)), desc='train result', leave=False):
    #     image_path = train_dataset.img_fns[i]
    #     image_name = os.path.basename(image_path)
    #     image = Image.open(image_path).convert("RGB")
    #     image = train_dataset.transform_img(image)
    #     forward_hook.features = []
    #     try:
    #         backbone(image[None].to(device))
    #     except patchcore.common.LastLayerToExtractReachedException:
    #         pass
    #     features = torch.cat(forward_hook.features, 0)  # b x 512 x h x w
    #
    #     print(features.shape)
    #     # def PCA_w(features,pca):
    #     #     b, c, h, w = features.shape
    #     #     features = (features.reshape(-1, features.shape[1])).cpu().numpy()
    #     #     features = torch.tensor(pca.transform(features))
    #     #     features = features.cuda()
    #     #     return features,b,h,w
    #     # features,b,h,w=PCA_w(features,pca)
    #     # features=features.reshape(b,n_component_pca,h,w)
    #
    #
    #     features = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1]).cpu().numpy()
    #     lda_predict = lda.transform(features)
    #     lda_predict = lda_predict.reshape(forward_hook.features[0].shape[2], forward_hook.features[0].shape[3])
    #     lda_predict = cv.resize(lda_predict, (image.shape[2], image.shape[1]))
    #     lda_predict = ndimage.gaussian_filter(lda_predict, sigma=gaussian_filter_sigma)
    #     # 将结果二值化然后求最大连通域的最小外接矩形。
    #     if lda_threshold is None:
    #         lda_mask = lda.predict(features).reshape(forward_hook.features[0].shape[2],
    #                                                  forward_hook.features[0].shape[3])
    #         lda_mask = cv.resize(lda_mask, (image.shape[2], image.shape[1]), interpolation=cv.INTER_NEAREST).astype(
    #             np.uint8) * 255
    #     else:
    #         lda_mask = (lda_predict > lda_threshold).astype(np.uint8) * 255
    #     # cv.imshow('graycsale image', lda_mask)
    #     cv.imwrite(os.path.join(cur_output_dir, 'mask'+image_name),lda_mask)
    #
    #
    #
    #
    #     cn, cc_labels, cc_stats, _ = cv.connectedComponentsWithStats(lda_mask, connectivity=8)
    #     mask = np.zeros_like(lda_predict, dtype=np.uint8)
    #     if cn > 1:
    #         max_cc_labels = np.argmax(np.array([i[-1] for i in cc_stats[1:]])) + 1
    #         y, x = np.where(cc_labels == max_cc_labels)
    #         rect = cv.minAreaRect(np.stack([x, y], axis=1))
    #         box = cv.boxPoints(rect)
    #         mask = cv.fillPoly(mask, box[None].astype(int), 255)
    #     plt.figure(figsize=(15, 15), dpi=180)
    #     plt.subplot(1, 3, 1)
    #     plt.imshow(image.permute(1, 2, 0) * torch.tensor(IMAGENET_STD) + torch.tensor(IMAGENET_MEAN))
    #     plt.subplot(1, 3, 2)
    #     plt.imshow(lda_predict, plt.cm.hot)
    #     plt.subplot(1, 3, 3)
    #     plt.imshow(mask, 'gray')
    #     plt.savefig(os.path.join(cur_output_dir, image_name))
    #     plt.close()
    #     # cv.imwrite(os.path.join(cur_output_dir, image_name), mask)
    #
    # # 测试数据集
    # for an in tqdm(os.listdir(os.path.join(train_dataset.root, train_dataset.classname, 'test')), 'test result'):
    #     test_dataset = MVTecDataset(train_dataset.root, train_dataset.classname, train_dataset.resize, split='test',
    #                                 anomaly=an)
    #     cur_output_dir = os.path.join(cur_classname_output_dir, 'test', f'{an}')
    #     os.makedirs(cur_output_dir, exist_ok=True)
    #     for i in tqdm(range(len(test_dataset)), desc=f'test {an} result', leave=False):
    #         image_path = test_dataset.img_fns[i]
    #         image_name = os.path.basename(image_path)
    #         image = Image.open(image_path).convert("RGB")
    #         image = test_dataset.transform_img(image)
    #         forward_hook.features = []
    #         try:
    #             backbone(image[None].to(device))
    #         except patchcore.common.LastLayerToExtractReachedException:
    #             pass
    #         features = torch.cat(forward_hook.features, 0)  # b x 512 x h x w
    #
    #
    #
    #
    #         features = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1]).cpu().numpy()
    #         lda_predict = lda.transform(features)
    #         lda_predict = lda_predict.reshape(forward_hook.features[0].shape[2], forward_hook.features[0].shape[3])
    #         lda_predict = cv.resize(lda_predict, (image.shape[2], image.shape[1]))
    #         lda_predict = ndimage.gaussian_filter(lda_predict, sigma=gaussian_filter_sigma)
    #
    #         # 将结果二值化然后求最大连通域的最小外接矩形。
    #         if lda_threshold is None:
    #             lda_mask = lda.predict(features).reshape(forward_hook.features[0].shape[2],
    #                                                      forward_hook.features[0].shape[3])
    #             lda_mask = cv.resize(lda_mask, (image.shape[2], image.shape[1]), interpolation=cv.INTER_NEAREST).astype(
    #                 np.uint8) * 255
    #         else:
    #             lda_mask = (lda_predict > lda_threshold).astype(np.uint8) * 255
    #         cv.imwrite(os.path.join(cur_output_dir, 'mask' + image_name), lda_mask)
    #         cn, cc_labels, cc_stats, _ = cv.connectedComponentsWithStats(lda_mask, connectivity=8)
    #         mask = np.zeros_like(lda_predict, dtype=np.uint8)
    #         if cn > 1:
    #             max_cc_labels = np.argmax(np.array([i[-1] for i in cc_stats[1:]])) + 1
    #             y, x = np.where(cc_labels == max_cc_labels)
    #             rect = cv.minAreaRect(np.stack([x, y], axis=1))
    #             box = cv.boxPoints(rect)
    #             mask = cv.fillPoly(mask, box[None].astype(int), 255)
    #         plt.figure(figsize=(15, 15), dpi=180)
    #         plt.subplot(1, 3, 1)
    #         plt.imshow(image.permute(1, 2, 0) * torch.tensor(IMAGENET_STD) + torch.tensor(IMAGENET_MEAN))
    #         plt.subplot(1, 3, 2)
    #         plt.imshow(lda_predict, plt.cm.hot)
    #         plt.subplot(1, 3, 3)
    #         plt.imshow(mask, 'gray')
    #         plt.savefig(os.path.join(cur_output_dir, image_name))
    #         plt.close()
    #         # cv.imwrite(os.path.join(cur_output_dir, image_name), mask)
    #
    #
    #
    #
    #         gc.collect()
    #         torch.cuda.empty_cache()
