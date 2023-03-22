import os

import cv2
import numpy as np
import torch.utils.data
from torchvision import models

import patchcore
from patchcore import patchcore as patchcore_model
import matplotlib.pylab as plt
import matplotlib.cm as cm


def _dummy_features(number_of_examples, shape_of_examples):
    return torch.Tensor(
        np.stack(number_of_examples * [np.ones(shape_of_examples)], axis=0)
    )


def _dummy_constant_dataloader(number_of_examples, shape_of_examples):
    features = _dummy_features(number_of_examples, shape_of_examples)
    return torch.utils.data.DataLoader(features, batch_size=4)


def _dummy_various_features(number_of_examples, shape_of_examples):
    images = torch.ones((number_of_examples, *shape_of_examples))
    multiplier = torch.arange(number_of_examples) / float(number_of_examples)
    for _ in range(images.ndim - 1):
        multiplier = multiplier.unsqueeze(-1)
    return multiplier * images


def _dummy_various_dataloader(path):
    img_path=os.listdir(path)
    imgs=[]
    for i in range(len(img_path)):
        img=cv2.resize(cv2.imread(os.path.join(path,img_path[i])),(112,112))
        imgs.append(img)
    imgs=np.array(imgs)
    imgs=torch.tensor(imgs)

    features =imgs.permute(0,3,1,2)
    return torch.utils.data.DataLoader(features, batch_size=1)


def _dummy_images(number_of_examples, image_shape):
    torch.manual_seed(0)
    return torch.rand([number_of_examples, *image_shape])


def _dummy_image_random_dataloader(number_of_examples, image_shape):
    images = _dummy_images(number_of_examples, image_shape)
    return torch.utils.data.DataLoader(images, batch_size=4)


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
    )
    return patchcore_instance


def _load_patchcore_from_path(load_path):
    patchcore_instance = patchcore_model.PatchCore(torch.device("cpu"))
    patchcore_instance.load_from_path(
        load_path=load_path,
        device=torch.device("cpu"),
        prepend="temp_patchcore",
        nn_method=patchcore.common.FaissNN(False, 4),
    )
    return patchcore_instance


def _approximate_greedycoreset_sampler_with_reduction(
        sampling_percentage, johnsonlindenstrauss_dim
):
    return patchcore.sampler.ApproximateGreedyCoresetSampler(
        percentage=sampling_percentage,
        device=torch.device("cpu"),
        number_of_starting_points=10,
        dimension_to_project_features_to=johnsonlindenstrauss_dim,
    )

def get_mask(root,classname):
    import os
    import cv2
    import torch
    import numpy as np
    from PIL import Image
    from tqdm import tqdm
    import patchcore.common
    import patchcore.backbones
    from torchvision import transforms
    from sklearn.cluster import KMeans
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from patchcore.datasets.mvtec import _CLASSNAMES, IMAGENET_MEAN, IMAGENET_STD
    from sklearn.semi_supervised import LabelPropagation
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


    backbone_name = 'wideresnet50'
    layer = 'layer2'
    device = 'cuda'
    # train
    data_root = '/media/wwkkb/0CCD76FCF63D1C29/wwkkb/MVTec'
    kmeans_f_rate = 0.01
    rate_unlabel = 0.02

    lda_f_num = 500
    output_dir = f'log/S3VM_patchcore{backbone_name}_{layer}'
    foreground_ratio = 0.2
    background_ratio = -3
    lda_threshold = None  # None自动
    gaussian_filter_sigma = 10
    n_clusters = 4

    object_classnames = ['carpet', 'grid', 'leather', 'tile', 'wood',  ]
    CLASS_NAMES = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
                         'hazelnut', 'leather','metal_nut', 'pill', 'screw', 'tile',
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
        train_dataset = MVTecDataset(root,classname, resize=resize)
        for image in tqdm(train_dataset, desc='extract feature'):
            try:
                backbone(image[None].to(device))
            except patchcore.common.LastLayerToExtractReachedException:
                pass
        features = forward_hook.features
        features = torch.cat(features, 0)  # b x 512 x h x w
        image_features = features.permute(0, 2, 3, 1).cpu().numpy()  # b x h x w x 512

        print('kmeans')
        kmeans.fit(image_features.reshape(-1, features.shape[1])[
                       np.random.permutation(image_features.reshape(-1, features.shape[1]).shape[0])[
                       :(kmeans_f_rate * np.sum(image_features.shape)).astype(int)]])
        # labels = kmeans.labels_
        print('kmeans predict')
        os.makedirs(os.path.join(output_dir, classname,),exist_ok=True)
        # Save the kmeans model
        pickle.dump(kmeans, open(os.path.join(output_dir, classname, "kmeans.dat"), "wb"))
        # kmeans = pickle.load(open(os.path.join(output_dir ,classname, "kmeans.dat"), "rb"))
        labels = kmeans.predict(image_features.reshape(-1, features.shape[1]))
        labels_imgs = labels.reshape(len(forward_hook.features), forward_hook.features[0].shape[2],
                                     forward_hook.features[0].shape[3])
        if background_ratio < 0:
            background_ratio = -background_ratio / labels_imgs.shape[1]
        # 以ratio作为边界框，选取边界框到边界的值统计hist
        background_mask = np.zeros(
            (len(forward_hook.features), forward_hook.features[0].shape[2], forward_hook.features[0].shape[3]),
            dtype=bool)
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
            (len(forward_hook.features), forward_hook.features[0].shape[2], forward_hook.features[0].shape[3]),
            dtype=bool)
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

        def test_LabelPropagation(*data):
            X, y, unlabeled_data_all = data
            b,h,w,c=unlabeled_data_all.shape
            unlabeled_data_all=unlabeled_data_all.reshape(-1,c)
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
            div = b
            indexs_=[]
            masks_=[]
            devide = int(len(unlabeled_data_all) / div)
            for i in range(div):
                pre = unlabeled_data_all[i * devide:(i + 1) * devide]
                label = clf.predict(pre)
                label=cv2.resize(label.reshape(h,w),(14,14)).flatten()
                index_ = np.array(np.where(label >0))[0]
                indexs_.append(index_+i*196)
                masks_.append(label[index_])
            # return unlabeled_data_all[:sum], clf #train
            return indexs_,masks_, clf  # debug

            # 获取预测准确率
            # print('Accuracy:%f' % clf.score(X[unlabeled_indices], true_labels))

        labeled_data = np.vstack((background_features, foreground_features))
        labeled_label = np.hstack((background_label, foreground_label))
        indexs_,masks_, clf = test_LabelPropagation(labeled_data, labeled_label,
                                            image_features)
        # 保存模型,我们想要导入的是模型本身，所以用“wb”方式写入，即是二进制方式,DT是模型名字
        # Save the kmeans model
        pickle.dump(clf, open(os.path.join(output_dir, classname, "dtr.dat"), "wb"))
        return indexs_,masks_, clf
def get_feature(root,classname,clf):
    import os
    import cv2
    import torch
    import numpy as np
    from PIL import Image
    from tqdm import tqdm
    import patchcore.common
    import patchcore.backbones
    from torchvision import transforms
    from sklearn.cluster import KMeans
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from patchcore.datasets.mvtec import _CLASSNAMES, IMAGENET_MEAN, IMAGENET_STD
    from sklearn.semi_supervised import LabelPropagation
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

    backbone_name = 'wideresnet50'
    layer = 'layer2'
    device = 'cuda'
    # train
    data_root = '/media/wwkkb/0CCD76FCF63D1C29/wwkkb/MVTec'
    kmeans_f_rate = 0.01
    rate_unlabel = 0.02

    lda_f_num = 500
    output_dir = f'log/S3VM_patchcore{backbone_name}_{layer}'
    foreground_ratio = 0.2
    background_ratio = -3
    lda_threshold = None  # None自动
    gaussian_filter_sigma = 10
    n_clusters = 4

    object_classnames = ['carpet', 'grid', 'leather', 'tile', 'wood', ]
    CLASS_NAMES = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile',
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
        test_dataset = MVTecDataset(root, classname, resize, split='test',
                                    anomaly='broken_large')
        # train_dataset = MVTecDataset(root, classname, resize=resize)
        for image in tqdm(test_dataset, desc='extract feature'):
            try:
                backbone(image[None].to(device))
            except patchcore.common.LastLayerToExtractReachedException:
                pass
        features = forward_hook.features
        features = torch.cat(features, 0)  # b x 512 x h x w
        image_features = features.permute(0, 2, 3, 1).cpu().numpy()  # b x h x w x 512
        b,h,w,c=image_features.shape
        mask=clf.predict(image_features.reshape(-1,image_features.shape[3]))
        mask=mask.reshape(b,h,w)
        mask=[cv2.resize(i,(112,112)) for i in mask]
        return mask
def test_patchcore_real_data():
    image_dimension = 112
    sampling_percentage = 0.1
    model = _standard_patchcore(image_dimension)
    model.sampler = _approximate_greedycoreset_sampler_with_reduction(
        sampling_percentage=sampling_percentage,
        johnsonlindenstrauss_dim=64,
    )
    root=r'/media/wwkkb/0CCD76FCF63D1C29/wwkkb/MVTec/'
    path=r'/media/wwkkb/0CCD76FCF63D1C29/wwkkb/MVTec/bottle/train/good3'
    test_path=r'/media/wwkkb/0CCD76FCF63D1C29/wwkkb/MVTec/bottle/test/broken_small'
    class_name='bottle'
    num_dummy_train_images = 50
    training_dataloader = _dummy_various_dataloader(path)
    # model.anomaly_scorer = patchcore.common.NearestNeighbourScorer(
    #     n_nearest_neighbours=3, nn_method=patchcore.common.ApproximateFaissNN(False, 4)
    # )
    indexs_,masks_, clf=get_mask(root,class_name)
    indexs_=np.concatenate(indexs_,0)
    masks_=np.concatenate(masks_,0)
    model.fit((training_dataloader,indexs_,masks_))

    num_dummy_test_images = 5
    test_dataloader = _dummy_various_dataloader(test_path)
    masks_svm=get_feature(root,class_name,clf)
    scores, masks, labels_gt, masks_gt = model.predict((test_dataloader,clf))
    for i in range(len(masks)):
        plt.subplot(1, 1, 1)
        plt.imshow(masks[i]*masks_svm[i], cmap=cm.hot)
        # os.makedirs(cur_output_dir, exist_ok=True)
        # plt.savefig(os.path.join(cur_output_dir, "{:0>3}".format(str(i)) + '.png'))
        plt.colorbar()
        plt.show()
    for mask, mask_gt in zip(masks, masks_gt):
        assert np.all(mask.shape == (image_dimension, image_dimension))
        assert np.all(mask_gt.shape == (image_dimension, image_dimension))

    # assert len(scores) == 5
