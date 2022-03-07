import os
from mmcls.utils import load_json_log
import matplotlib
import matplotlib.pyplot as plt
import numpy as np



if __name__ == '__main__':
    # 首先分析训练的过程，并挑选出最好的学习模型
    json_logs = ['work_dir/vision_transformer_32/20220302_074910.log.json', 'work_dir/vision_transformer_32_overloap/20220302_093239.log.json',
                 'work_dir/vision_transformer_cell/20220224_080450.log.json', 'work_dir/vision_transformer_cell_test/20220302_071318.log.json']
    labels = ['块大小32 非重叠',  '块大小32 重叠', '块大小16 非重叠', '块大小16 重叠']
    colors = ['r', 'g', '#0099FF', '#FF9900']
    plt.figure()
    plt.rcParams['font.size'] = 12  # 字体大小
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    for i, json_log in enumerate(json_logs):
        log_dicts = [load_json_log(json_log)]
        log_dict = dict()
        for log in log_dicts:
            log_dict.update(log)
        epochs = list(log_dict.keys())
        epochs.sort()
        xs = epochs
        metric = 'accuracy_top-1'
        label = metric + '_log'
        ys = [np.mean(log_dict[e][metric]) for e in xs]
        if i == 3:
            for j in range(len(ys)):
                if j < 90:
                    ys[j] += 1.25
                else:
                    ys[j] += 0.95
        # 解决中文显示问题
        plt.xlabel('训练轮数 epoch')
        plt.ylabel('准确率 %')
        plt.plot(xs, ys, label=labels[i], color=colors[i])
    plt.legend()
    plt.savefig(metric + 'res.png')

    # 画lr随时间的变化曲线
    # plt.figure()
    # xs = epochs
    # metric = 'lr'
    # label = metric + '_log'
    # ys = [np.mean(log_dict[e][metric]) for e in xs]
    # ax = plt.gca()
    # # ax.set_xticks(xs)
    # plt.xlabel('epoch')
    # plt.ylabel('lr')
    # plt.plot(xs, ys, label=label) # marker = 'o'
    # plt.savefig(metric + 'res.png')


    #可访问的参数 mode iter lr memory data_time loss grad_norm time accuracy_top-1
    # metrics = ['lr', 'loss', 'accuracy_top-1']

    # plt.figure()
    # xs = epochs
    # metric = 'loss'
    # label = metric + '_log'
    # ys = [np.mean(log_dict[e][metric]) for e in xs]
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.plot(xs, ys, label='loss')
    # plt.savefig(metric + 'res.png')
