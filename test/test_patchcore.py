import os

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


def _dummy_various_dataloader(number_of_examples, shape_of_examples):
    features = _dummy_various_features(number_of_examples, shape_of_examples)
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


def test_dummy_patchcore1():
    import cv2
    output_dir = f'log/test_pa'
    data_path=r'/home/wwkkb/MVTec'
    data_mask_path=r'/home/wwkkb/cl_wideresnet50_layer2'
    os.makedirs(output_dir, exist_ok=True)
    image_dimension = 112
    model = _standard_patchcore(image_dimension)

    object_classnames = ['carpet', 'grid', 'leather', 'tile', 'wood']
    CLASS_NAMES = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile',
        'toothbrush', 'transistor', 'wood', 'zipper'
    ]

    for classname in CLASS_NAMES:
        # for classname in ['screw']:
        if classname in object_classnames:
            continue
        cur_classname_output_dir = os.path.join(output_dir, classname)

        path = os.path.join(data_path,classname,'train/good3')
        path_mask = os.path.join(data_mask_path,classname,'train/good3')

        dir = os.listdir(path)
        images=np.ones((len(dir),112,112,3),dtype=np.uint8)
        i=0
        for img in dir:
            image=cv2.imread(os.path.join(path, img))
            mask=cv2.imread(os.path.join(path_mask,'mask'+img))
            image,mask=cv2.resize(image,(112,112)),cv2.resize(mask,(112,112))/255
            images[i]=np.uint8(image*mask)

            # x=images[i]
            # cv2.imshow('x',np.uint8(x))
            # cv2.waitKey(0)
            i += 1
        # os.makedirs(os.path.join('/home/wwkkb/result',classname),exist_ok=True)
        # cv2.imwrite(os.path.join('/home/wwkkb/result',classname,'good0.png'),images[0])
        # cv2.imwrite(os.path.join('/home/wwkkb/result',classname,'good1.png'),images[1])
        # cv2.imwrite(os.path.join('/home/wwkkb/result',classname,'good2.png'),images[2])
        # images = np.stack(cv2.resize(cv2.imread(os.path.join(path, img)), (112, 112)) for img in dir)
        images=torch.from_numpy(images)
        images = images.permute(0, 3, 1, 2)
        training_dataloader = torch.utils.data.DataLoader(images, batch_size=64)
        print(model.featuresampler)
        model.fit(training_dataloader)
        test_path=os.path.join(data_path,classname,'test')
        broken_list=sorted(os.listdir(test_path))
        for i in range(len(broken_list)):
            broken=broken_list[i]
            path = os.path.join(data_path, classname, 'test',broken)
            path_mask = os.path.join(data_mask_path, classname, 'test',broken)
            dir = os.listdir(path)
            images = np.ones((len(dir), 112, 112, 3), dtype=np.uint8)
            i = 0
            for img in dir:
                image = cv2.imread(os.path.join(path, img))
                mask = cv2.imread(os.path.join(path_mask, 'mask' + img))
                image, mask = cv2.resize(image, (112, 112)), cv2.resize(mask, (112, 112)) / 255
                images[i] = np.uint8(image * mask)
                i+=1
            # os.makedirs(os.path.join('/home/wwkkb/result', classname,broken),exist_ok=True)
            # cv2.imwrite(os.path.join('/home/wwkkb/result', classname, broken,'000.png'), images[0])
            # cv2.imwrite(os.path.join('/home/wwkkb/result', classname, broken,'001.png'), images[1])


            # test_features =np.stack(cv2.resize(cv2.imread(os.path.join(path_test, img)), (112, 112)) for img in dir[:20])
            # mask_=cv2.imread('/home/wwkkb/cl_wideresnet50_layer2/bottle/test/broken_small/mask001.png')/255
            # test_features[1]=cv2.resize(mask_,(112,112))*test_features[1]
            # for _ in test_features:
            #
            #     x=_
            #     cv2.imshow('x',np.uint8(x))
            #     cv2.waitKey(0)
            test_features=images
            test_features=torch.from_numpy(test_features)
            test_features=test_features.permute(0,3,1,2)
            scores, masks = model.predict(test_features)
            for i in range(len(masks)):
                plt.imshow(masks[i], cmap=cm.hot)
                os.makedirs(os.path.join('/home/wwkkb/Mvtec_result',classname,'test',broken),exist_ok=True)
                plt.savefig(os.path.join('/home/wwkkb/Mvtec_result',classname,'test',broken,str(i)+'.png'))
                plt.colorbar()
                plt.show()

            assert all([score > 0 for score in scores])
            for mask in masks:
                assert np.all(mask.shape == (image_dimension, image_dimension))



def test_dummy_patchcore():
    image_dimension = 112
    model = _standard_patchcore(image_dimension)
    training_dataloader = _dummy_constant_dataloader(
        20, [3, image_dimension, image_dimension]
    )
    print(model.featuresampler)
    model.fit(training_dataloader)

    test_features = torch.Tensor(2 * np.ones([2, 3, image_dimension, image_dimension]))
    scores, masks = model.predict(test_features)

    assert all([score > 0 for score in scores])
    for mask in masks:
        assert np.all(mask.shape == (image_dimension, image_dimension))


def test_patchcore_on_dataloader():
    """Test PatchCore on dataloader and assure training scores are zero."""
    image_dimension = 112
    model = _standard_patchcore(image_dimension)

    training_dataloader = _dummy_constant_dataloader(
        4, [3, image_dimension, image_dimension]
    )
    model.fit(training_dataloader)
    scores, masks, labels_gt, masks_gt = model.predict(training_dataloader)

    assert all([score < 1e-3 for score in scores])
    for mask, mask_gt in zip(masks, masks_gt):
        assert np.all(mask.shape == (image_dimension, image_dimension))
        assert np.all(mask_gt.shape == (image_dimension, image_dimension))


def test_patchcore_load_and_saveing(tmpdir):
    image_dimension = 112
    model = _standard_patchcore(image_dimension)

    training_dataloader = _dummy_constant_dataloader(
        4, [3, image_dimension, image_dimension]
    )
    model.fit(training_dataloader)
    model.save_to_path(tmpdir, "temp_patchcore")

    test_features = torch.Tensor(
        1.234 * np.ones([2, 3, image_dimension, image_dimension])
    )
    scores, masks = model.predict(test_features)
    other_scores, other_masks = model.predict(test_features)

    assert np.all(scores == other_scores)
    for mask, other_mask in zip(masks, other_masks):
        assert np.all(mask == other_mask)


def test_patchcore_real_data():
    image_dimension = 112
    sampling_percentage = 0.1
    model = _standard_patchcore(image_dimension)
    model.sampler = _approximate_greedycoreset_sampler_with_reduction(
        sampling_percentage=sampling_percentage,
        johnsonlindenstrauss_dim=64,
    )

    num_dummy_train_images = 50
    training_dataloader = _dummy_various_dataloader(
        num_dummy_train_images, [3, image_dimension, image_dimension]
    )
    model.fit(training_dataloader)

    num_dummy_test_images = 5
    test_dataloader = _dummy_various_dataloader(
        num_dummy_test_images, [3, image_dimension, image_dimension]
    )
    scores, masks, labels_gt, masks_gt = model.predict(test_dataloader)

    for mask, mask_gt in zip(masks, masks_gt):
        assert np.all(mask.shape == (image_dimension, image_dimension))
        assert np.all(mask_gt.shape == (image_dimension, image_dimension))

    assert len(scores) == 5
