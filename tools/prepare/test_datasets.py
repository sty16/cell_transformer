import os


if __name__ == '__main__':
    train_path = '/Data/JOBE/imagenet/train/'
    # train_path = '/Data/JOBE/data_/'
    arr = []
    folders = []
    for folder in os.listdir(train_path):
        fns = os.listdir(os.path.join(train_path, folder))
        arr.append(len(fns))
        folders.append(folder)
    print(arr)
    print(sum(arr))
    print(folders)


