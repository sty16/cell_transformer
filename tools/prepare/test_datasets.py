import os


if __name__ == '__main__':
    train_path = '/Data/JOBE/imagenet/train/'
    arr = []
    for folder in os.listdir(train_path):
        fns = os.listdir(os.path.join(train_path, folder))
        arr.append(len(fns))
    print(arr)
    print(sum(arr))


