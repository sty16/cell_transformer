import os
from mmcls.utils import load_json_logs
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    file_dirs = './work_dir/vit'
    files = os.listdir(file_dirs)
    json_logs = [os.path.join(file_dirs, f) for f in os.listdir(file_dirs) if f.endswith('.json')]
    log_dicts = load_json_logs(json_logs)
    log_dict = dict()
    for i, log in enumerate(log_dicts):
        log_dict.update(log)
    epochs = list(log_dict.keys())
    epochs.sort()

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

    plt.figure()
    xs = epochs
    metric = 'accuracy_top-1'
    label = metric + '_log'
    ys = [np.mean(log_dict[e][metric]) for e in xs]
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.plot(xs, ys, label='accuracy')
    plt.savefig(metric + 'res.png')
    #可访问的参数 mode iter lr memory data_time loss grad_norm time accuracy_top-1
    metrics = ['lr', 'loss', 'accuracy_top-1']

    plt.figure()
    xs = epochs
    metric = 'loss'
    label = metric + '_log'
    ys = [np.mean(log_dict[e][metric]) for e in xs]
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(xs, ys, label='loss')
    plt.savefig(metric + 'res.png')
