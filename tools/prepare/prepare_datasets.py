import os
import shutil
from sklearn.model_selection import train_test_split, KFold
import glob
import cv2


class Data2Imagenet:
    def __init__(self, saved_imagenet_path):
        self.ann = []
        self.saved_imagenet_path = saved_imagenet_path
        self.under_sample = 1000
        self.aug = {'NGB', 'EOS', 'BAS', 'PMO', 'MYB', 'EBO'}

    def to_imagenet(self, root):
        folders = [
            d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
        ]
        folders.sort()
        folder_to_idx = {folders[i]: i for i in range(len(folders))}
        if not os.path.exists(os.path.join(saved_imagenet_path, 'train')):
            os.makedirs(os.path.join(saved_imagenet_path, 'train'))
        if not os.path.exists(os.path.join(saved_imagenet_path, 'val')):
            os.makedirs(os.path.join(saved_imagenet_path, 'val'))
        if not os.path.exists(os.path.join(saved_imagenet_path, 'meta')):
            os.makedirs(os.path.join(saved_imagenet_path, 'meta'))
        for folder_name in folders:
            _dir = os.path.join(root, folder_name)
            fns = glob.glob(_dir + "/*.tiff")
            # 如果类别数量过大，则进行欠采样
            if len(fns) > self.under_sample:
                fns = fns[:self.under_sample]
            label = folder_to_idx[folder_name]
            train_fns, val_fns = train_test_split(fns, test_size=0.20)
            # 是否对数据进行增强
            flag_aug = folder_name in self.aug
            for fn in train_fns:
                if not os.path.exists(os.path.join(saved_imagenet_path, 'train', folder_name)):
                    os.makedirs(os.path.join(saved_imagenet_path, 'train', folder_name))
                filename = fn.split('/')[-1]
                file_name = os.path.basename(filename)
                target_file = os.path.join(f'{self.saved_imagenet_path}/train/{folder_name}/', file_name[:-5] + '.jpg')
                img = cv2.imread(fn)
                cv2.imwrite(target_file, img)
                # 进行数据增强
                if flag_aug:
                    imgs = self.data_augment(img)
                    if folder_name == 'EOS':
                        imgs = imgs[:2]
                    for i, img in enumerate(imgs):
                        target_file = os.path.join(f'{self.saved_imagenet_path}/train/{folder_name}/', file_name[:-5] + '_aug_' + str(i) + '.jpg')
                        cv2.imwrite(target_file, img)
            for fn in val_fns:
                filename = fn.split('/')[-1]
                file_name = os.path.basename(filename)
                self.ann.append((file_name[:-5] + '.jpg', label))
                target_file = os.path.join(f'{self.saved_imagenet_path}/val/', file_name[:-5] + '.jpg')
                img = cv2.imread(fn)
                cv2.imwrite(target_file, img)
                # 进行数据增强
                if flag_aug:
                    imgs = self.data_augment(img)
                    if folder_name == 'EOS':
                        imgs = imgs[:2]
                    for i, img in enumerate(imgs):
                        self.ann.append((file_name[:-5] + '_aug_' + str(i) + '.jpg', label))
                        target_file = os.path.join(f'{self.saved_imagenet_path}/val/', file_name[:-5] + '_aug_' + str(i) + '.jpg')
                        cv2.imwrite(target_file, img)
        self.save_imagenet_txt()

    def save_imagenet_txt(self):
        self.ann.sort()
        val_file = os.path.join(self.saved_imagenet_path, 'meta', 'val.txt')
        with open(val_file, 'w') as f:
            for fn, label in self.ann:
                f.write(f'{fn} {label}\n')

    def to_kfold_imagenet(self, root):
        folders = [
            d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
        ]
        folders.sort()
        folder_to_idx = {folders[i]: i for i in range(len(folders))}
        kfold = 5
        ann = [[] for _ in range(kfold)]
        for i in range(kfold):
            saved_path = os.path.join(saved_imagenet_path, str(i))
            for folder in ['train', 'val', 'meta']:
                if not os.path.exists(os.path.join(saved_path, folder)):
                    os.makedirs(os.path.join(saved_path, folder))
        for folder_name in folders:
            _dir = os.path.join(root, folder_name)
            fns = glob.glob(_dir + "/*.jpg")
            label = folder_to_idx[folder_name]
            kf = KFold(n_splits=kfold)
            k_splits = kf.split(fns)
            for i, (train_ids, val_ids) in enumerate(k_splits):
                saved_path = os.path.join(saved_imagenet_path, str(i))
                if not os.path.exists(os.path.join(saved_path, 'train', folder_name)):
                    os.makedirs(os.path.join(saved_path, 'train', folder_name))
                for f_id in train_ids:
                    fn = fns[f_id]
                    shutil.copy(fn, f'{saved_path}/train/{folder_name}/')
                for f_id in val_ids:
                    fn = fns[f_id]
                    filename = fn.split('/')[-1]
                    ann[i].append((filename, label))
                    shutil.copy(fn, f'{saved_path}/val/')

        for i in range(kfold):
            saved_path = os.path.join(saved_imagenet_path, str(i))
            val_file = os.path.join(saved_path, 'meta', 'val.txt')
            with open(val_file, 'w') as f:
                for fn, label in ann[i]:
                    f.write(f'{fn} {label}\n')

    def data_augment(self, img):
        # 水平翻转
        img_0 = cv2.flip(img, 1)
        # 垂直翻转
        img_1 = cv2.flip(img, 0)
        # 旋转90度
        img_2 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        # 旋转180度
        img_3 = cv2.rotate(img, cv2.ROTATE_180)
        return [img_0, img_1, img_2, img_3]

if __name__ == '__main__':
    data_path = '/Data/JOBE/data_/'
    saved_imagenet_path = '/Data/JOBE/imagenet/'
    data2imagenet = Data2Imagenet(saved_imagenet_path)
    data2imagenet.to_imagenet(data_path)







