import argparse
import collections
import copy
import os.path
import warnings
import pickle
import shutil

import matplotlib.pyplot as plt
import mmcv
import torch
from mmcv import DictAction
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.parallel import MMDataParallel
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier

# hook在不改变模型结构的情况下，查看模块输出的特征图

class FeatureHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.inputs = []
        self.features = []

    def hook_fn(self, module, input, output):
        self.inputs.append(copy.deepcopy(input))
        self.features.append(copy.deepcopy(output))

    def remove(self):
        self.hook.remove

def plot_embedding(data, gt_label, gt_classes, save_file, title):
    min_p, max_p = np.min(data, axis=0), np.max(data, axis=0)
    data = (data - min_p) / (max_p - min_p)
    g = sns.relplot(x=data[:, 0], y=data[:, 1], hue=gt_classes, kind='scatter')
    # ax = sns.scatterplot(data[:, 0], data[:, 1], hue=gt_classes)
    ax = g.axes[0][0]
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    g.savefig(save_file)


def get_class_accuracy(pred, gt_labels, classes, topk=1, thrs=0.):
    pred_labels = np.argmax(pred, axis=1)
    res_precision = precision_score(gt_labels, pred_labels, average=None)
    res_recall = recall_score(gt_labels, pred_labels, average=None)
    res_recall = [round(x, 5) for x in res_recall]
    res_precision = [round(x, 5) for x in res_precision]
    eval_result = {}
    assert len(res_recall) == len(classes)
    # eval_result['f1_score'] = {k: v for k, v in zip(classes, res_f1_score)}
    eval_result['precision'] = {k: v for k, v in zip(classes, res_precision)}
    eval_result['recall   '] = {k: v for k, v in zip(classes, res_recall)}
    print("平均分类准确率:%")
    # TODO 计算平均的分类准确率
    acc_array = (pred_labels == gt_labels)
    acc = (sum(acc_array) / acc_array.size)
    print(acc)
    for metrics, value in eval_result.items():
        print(metrics, value)
    return eval_result

def plot_feature_heatmap(data, save_path):
    C, H, W = data.shape
    data = torch.mean(data, dim=0)
    feature = data.cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig = sns.heatmap(feature, ax=ax)
    ax.set_axis_off()
    plt.savefig(save_path)
    plt.close('all')

def plot_image(filename, save_file):
    im = plt.imread(filename)
    plt.imshow(im)
    plt.savefig(save_file)

# 查看输出特征图
def get_feature(backbone_hook, saved_path):
    tokens = backbone_hook.features
    inputs = []
    for input in backbone_hook.inputs:
        # 去除tensor的包裹
        inputs.append(input[0])
    inputs = torch.concat(inputs, dim=0)
    feature_maps = []
    for token in tokens:
        x = token[-1]  # 取消tuple的包裹
        if isinstance(x, list):
            feature_map, _ = x
            feature_maps.append(feature_map)
        else:
            assert 'error x is not a list'
    feature_maps = torch.concat(feature_maps, dim=0)
    B = feature_maps.shape[0]
    root_dir = os.path.join(saved_path, 'heatmap')
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    for i in range(B):
        heatmap_file = 'heatmap' + str(i) + '.png'
        save_path = os.path.join(root_dir, heatmap_file)
        plot_feature_heatmap(feature_maps[i, :], save_path)


def get_tsne(backbone_hook, gt_labels, gt_classes, saved_path):
    tokens = backbone_hook.features
    inputs = []
    for input in backbone_hook.inputs:
        # 去除tensor的tuple包裹
        inputs.append(input[0])
    inputs = torch.concat(inputs, dim=0)
    B = inputs.shape[0]
    inputs = inputs.reshape(B, -1).contiguous().cpu().numpy()
    tsne = TSNE(n_components=2)
    tsne.fit_transform(inputs)
    inputs_emb = tsne.embedding_
    title = ''
    root_dir = os.path.join(saved_path, 'tsne')
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    save_file = os.path.join(root_dir, 'inputs_tsne.png')
    plot_embedding(inputs_emb, gt_labels, gt_classes, save_file, title)

    cls_tokens = []
    for token in tokens:
        x = token[-1]  # 取消tuple的包裹
        if isinstance(x, list):
            _, cls_token = x
            cls_tokens.append(cls_token)
        else:
            assert 'error x is not a list'
    # features 网络输出的分类全连接前的向量
    features = torch.concat(cls_tokens, dim=0)
    # scores网络输出的各个分类结果的分数
    feature_X = features.cpu().numpy()
    tsne = TSNE(n_components=2)
    tsne.fit_transform(feature_X)
    tsne_emb = tsne.embedding_
    save_file = os.path.join(root_dir, 'cls_tsne.png')
    title = ''
    plot_embedding(tsne_emb, gt_labels, gt_classes, save_file, title)


def main(config, checkpoint_file):
    # CLASSES = ('Prim', 'Lym', 'Mono', 'Plas', 'Red', 'Promy', 'Myelo', 'Late', 'Rods', 'Lobu', 'Eosl')
    CLASSES = ('BAS', 'EBO', 'EOS', 'LYT', 'MON', 'MYB', 'MYO', 'NGB', 'NGS', 'PMO')
    cfg = mmcv.Config.fromfile(config)
    cfg.model.pretrained = None
    cfg.model.backbone.init_cfg = None
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False, shuffle=False,
        round_up=True)
    model = build_classifier(cfg.model)
    checkpoint = load_checkpoint(model, checkpoint_file, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    gt_labels = list(dataset.get_gt_labels())
    gt_classes = [CLASSES[x] for x in gt_labels]
    model.eval()
    results = []
    backbone = model.module.backbone
    backbone_hook = FeatureHook(backbone)
    dataset = data_loader.dataset
    img_metas = []
    meta_key = 'img_metas'
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)
            data_meta = data[meta_key].data
            img_metas += data_meta
            results.extend(result)
    filenames = [meta_info['filename'] for img_infos in img_metas for meta_info in img_infos]
    # 将结果序列化到pkl文件中
    res_infos = dict()
    res_infos['backbone_hook'] = backbone_hook
    res_infos['out'] = results
    res_infos['gt_labels'] = gt_labels
    res_infos['gt_classes'] = gt_classes
    res_infos['filenames'] = filenames
    res_infos['CLASSES'] = CLASSES
    mmcv.dump(res_infos, 'tmp.pkl')

def main1():
    module_name = 'vit-base-cell'
    res_infos = mmcv.load('tmp.pkl')
    backbone_hook = res_infos['backbone_hook']
    gt_labels = res_infos['gt_labels']
    gt_classes = res_infos['gt_classes']
    saved_path = os.path.join('./result/imgs/', module_name)
    # 对数据进行分析
    get_tsne(backbone_hook, gt_labels, gt_classes, saved_path)


if __name__ == '__main__':
    config = 'configs/vision_transformer/vit-base-cell-p16_pt-64xb64_in1k-224.py'
    checkpoint_file = 'work_dir/vision_transformer_cell_test/epoch_35.pth'
    main(config, checkpoint_file)
    main1()




























