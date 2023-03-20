import patchcore.common
import patchcore.backbones
import os
import gc
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import patchcore.common
import patchcore.backbones
from torchvision import transforms
from sklearn.cluster import KMeans
from patchcore.datasets.mvtec import _CLASSNAMES, IMAGENET_MEAN, IMAGENET_STD

import patchcore
import matplotlib.pylab as plt
import matplotlib.cm as cm
import pickle
import tracemalloc

tracemalloc.start()

snapshot1 = tracemalloc.take_snapshot()




resize = 640
# data_root = '/home/flyinghu/Data/mvtec_anomaly_detection'
data_root = '/media/wwkkb/0CCD76FCF63D1C29/wwkkb/MVTec'
backbone_name = 'wideresnet50'
layer = 'layer2'
device = 'cuda'
kmeans_f_rate = 0.01
# rate_unlabel=0.02
# lda_f_num = 5000
rate_unlabel = 0.005
lda_f_num = 200
output_dir = f'log/cl_1_good33_mask_unlabel{backbone_name}_{layer}'
# kmeans_f_num = 50000
# lda_f_num = 5000
foreground_ratio = 0.2
background_ratio = -3
lda_threshold = None  # None自动
gaussian_filter_sigma = 10
n_clusters = 4

object_classnames = ['carpet', 'grid', 'leather', 'tile', 'wood',
    'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', ]
CLASS_NAMES = ['pill', 'screw', 'tile',
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
# lda = LinearDiscriminantAnalysis()
forward_hook = ForwardHook('cuda')
os.makedirs(output_dir, exist_ok=True)
if isinstance(network_layer, torch.nn.Sequential):
    network_layer[-1].register_forward_hook(forward_hook)
elif isinstance(network_layer, torch.nn.Module):
    network_layer.register_forward_hook(forward_hook)

for classname in CLASS_NAMES:
    if classname in object_classnames:
        continue

    cur_classname_output_dir = os.path.join(output_dir, classname)
    os.makedirs(cur_classname_output_dir, exist_ok=True)
    # print(classname)
    train_dataset = MVTecDataset(data_root, classname, resize=resize)

    clf = pickle.load(open(os.path.join(output_dir, classname, "dtr.dat"), "rb"))

    # test
    dic = dict()
    x = os.listdir(os.path.join(train_dataset.root, train_dataset.classname, 'test'))
    for i in range(len(x)):
        if x[i] == 'good':
            x[i] = x[0]
            x[0] = 'good'
    aucs = []
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

            image_test = np.array(image).astype(np.uint8)
            # image_test = image
            # image_test = np.array(image_test).astype(np.uint8)
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
                x1=cv2.imread(cur_classname_mask, 0)
                mask_true_gray = cv2.resize(x1, shape_end)
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


        model = pickle.load(open(os.path.join(output_dir, classname, "patchcore.dat"), "rb"))
        scores, masks = model.predict((torch.from_numpy(features), shape_))


        # h=pow(len(label),0.5)
        def get_auc(mask_true_gray, mask,cur_output_dir):
            from sklearn.metrics import roc_curve, auc
            mask_true_gray = (np.concatenate(mask_true_gray)).flatten()
            mask = (np.concatenate(mask)).flatten()
            mask_ = (mask - mask.min()) / (mask.max() - mask.min())
            mask_true_gray = mask_true_gray.astype(int)
            mask_true_gray[0] = 1
            fpr, tpr, thresholds = roc_curve(mask_true_gray.flatten(), mask_.flatten(), pos_label=1)

            roc_auc = auc(fpr, tpr)
            plt.title('Receceiver Operating Characteristic')
            plt.plot(fpr, tpr, '#9400D3', label=u'AUC = %0.3f' % roc_auc)
            plt.legend(loc='lower right')
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([-0.1, 1.1])
            plt.ylim([-0.1, 1.1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.grid(linestyle='-.')
            plt.grid(True)
            # plt.show()
            plt.savefig(os.path.join(cur_output_dir, 'auc.png'))
            plt.close()
            print(roc_auc)
            # del plt
            return roc_auc


        dic[an] = get_auc(mask_true_grays, masks,cur_output_dir)

        _ = np.zeros((shape_[0], 112, 112))
        labels = labels.reshape((shape_[0], shape_[1], shape_[2]))
        for i in range(len(labels)):
            x1=cv2.resize(labels[i], (112, 112))
            _[i] = masks[i] * x1
        masks = _
        del _
        for i in range(len(labels)):
            # mask = np.where(mask < min_, min_, mask)

            # acc, max_, min_ = get_accuracy(masks[i], mask_true_grays[i])
            # min_s.append(min_)
            # max_s.append(max_)
            # acc_s.append(acc)
            # plt.figure(figsize=(30, 30), dpi=360)
            plt.subplot(1, 3, 1)
            x1=cv2.resize(mask_true_grays[i], masks[i].shape)
            plt.imshow(x1,cmap=plt.get_cmap('gray'))
            plt.subplot(1, 3, 2)
            x2=cv2.resize(image_test, masks[i].shape)
            plt.imshow(x2)
            plt.subplot(1, 3, 3)
            plt.imshow(masks[i], cmap=cm.hot)
            os.makedirs(cur_output_dir, exist_ok=True)

            plt.savefig(os.path.join(cur_output_dir, "{:0>3}".format(str(i)) + '.png'))
            plt.colorbar()
            # plt.show()
            plt.close()

    with open(os.path.join(output_dir, classname, 'aucs.txt'), 'w') as file:
        file.write(str(dic))
    del image_test,image
    del features
    del clf,masks
    del model
    gc.collect()
    torch.cuda.empty_cache()
    snapshot2 = tracemalloc.take_snapshot()

    top_stats = snapshot2.compare_to(snapshot1, 'lineno')

    print("[ Top 10 differences ]")
    for stat in top_stats[:10]:
        print(stat)
