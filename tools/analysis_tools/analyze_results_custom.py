import argparse
import collections
import copy
import warnings
import pickle

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


def plot_confusion_matrix(gt_labels, pred_labels):
    label = ['Prim', 'Lym', 'Mono', 'Plas', 'Red', 'Promy', 'Myelo', 'Late', 'Rods', 'Lobu', 'Eosl']
    c_mat = confusion_matrix(pred_labels, gt_labels)
    c_normalized = c_mat.astype('float') / c_mat.sum(axis=1)[:, np.newaxis]
    plt.figure()
    plt.imshow(c_normalized, cmap="viridis")
    num_label = len(label)
    indices = range(num_label)
    plt.xticks(indices, label, rotation=45)
    plt.yticks(indices, label)
    cb = plt.colorbar()
    cb.set_ticks(np.arange(0, 1.0, 0.1))
    plt.title('confusion matrix')
    for i in range(num_label):
        for j in range(num_label):
            plt.text(i, j, c_mat[i][j], va='center', ha='center')
    plt.savefig('./confusion1.jpg')
    plt.show()

def get_class_accuracy(pred, gt_labels, classes, topk=1, thrs=0.):
    pred_labels = np.argmax(pred, axis=1)
    res_precision = precision_score(gt_labels, pred_labels, average=None)
    res_recall = recall_score(gt_labels, pred_labels, average=None)
    res_recall = [round(x, 3) for x in res_recall]
    res_precision = [round(x, 3) for x in res_precision]
    eval_result = {}
    assert len(res_recall) == len(classes)
    # eval_result['f1_score'] = {k: v for k, v in zip(classes, res_f1_score)}
    eval_result['precision'] = {k: v for k, v in zip(classes, res_precision)}
    eval_result['recall   '] = {k: v for k, v in zip(classes, res_recall)}
    print("\n")
    for metrics, value in eval_result.items():
        print(metrics, value)
    return eval_result


def get_tsne(backbone_hook, gt_labels, gt_classes):
    tokens = backbone_hook.features
    inputs = []
    for input in backbone_hook.inputs:
        # 去除tensor的包裹
        inputs.append(input[0])
    inputs = torch.concat(inputs, dim=0)
    B = inputs.shape[0]
    inputs = inputs.reshape(B, -1).contiguous().cpu().numpy()
    tsne = TSNE(n_components=2)
    tsne.fit_transform(inputs)
    inputs_emb = tsne.embedding_
    title = 't-SNE embedding of the inputs'
    save_file = 'inputs_tsne.png'
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



def main():
    config = 'configs/vision_transformer/vitfg-base-p16_ft-64xb64_in1k-384.py'
    checkpoint_file = 'work_dir/vit_fine_grained/latest.pth'
    CLASSES = ('Prim', 'Lym', 'Mono', 'Plas', 'Red', 'Promy', 'Myelo', 'Late', 'Rods', 'Lobu', 'Eosl')
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
    # for name, module in model.named_modules():
    #     print(name, module)
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)
            batch_size = len(result)
            results.extend(result)
    get_tsne(backbone_hook, gt_labels, gt_classes)
    scores = np.asarray(results)
    pred_labels = np.argmax(scores, axis=1)
    get_class_accuracy(scores, gt_labels, CLASSES)
    plot_confusion_matrix(pred_labels, gt_labels)
    # 将结果序列化到pkl文件中
    # result = dict()
    # result['tsne_emb'] = tsne_emb
    # result['gt_labels'] = gt_labels
    # result['gt_classes'] = gt_classes
    # mmcv.dump(result, 'tmp.pkl')

def main1():
    result = mmcv.load('tmp.pkl')
    tsne_emb = result['tsne_emb']
    gt_labels = result['gt_labels']
    gt_classes = result['gt_classes']
    plot_embedding(tsne_emb, gt_labels, gt_classes, 'tsne_feature.png')

if __name__ == '__main__':
    main()




























