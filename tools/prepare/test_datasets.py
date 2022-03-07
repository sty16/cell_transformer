import os
import torch


if __name__ == '__main__':
    # train_path = '/Data/JOBE/imagenet/train/'
    # # train_path = '/Data/JOBE/data_/'
    # arr = []
    # folders = []
    # for folder in os.listdir(train_path):
    #     fns = os.listdir(os.path.join(train_path, folder))
    #     arr.append(len(fns))
    #     folders.append(folder)
    arr = torch.randn((10))
    for val in arr:
        print(val.item())
    # val, idx = torch.sort(arr)
    # s = torch.sum(val, dim=1, keepdim=True)
    # val /= s
    # s1 = torch.cumsum(val, dim=1)
    # print(val)
    # print(s1)
    #


