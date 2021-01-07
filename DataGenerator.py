import os
import numpy as np
import cv2

from cv2 import resize, imread, imshow, waitKey  # BGR
from keras.utils import Sequence
from util import utils

import imgaug as ia
from imgaug import augmenters as iaa


class DataGenerator(Sequence):
    '''
    Argument:
        data_file: 保存数据集的文本文件路径
        data_dir: 保存数据集文件夹
        prob:       list 用于划分训练集
    '''

    def __init__(self, data_file, data_dir, img_shape=(128, 128), batch_size=16, data_aug=False, prob=None,
                 shuffle=True):
        self.data_dir = data_dir
        self.img_shape = img_shape
        self.batch_size = batch_size
        self.data_aug = data_aug
        self.shuffle = shuffle

        # 数据增强器
        self.aug_pipe = self.augmenter()

        # 数据字典
        self.data_list = []
        self.data_map = {}

        # 从文件中读取data_list
        with open(data_file, 'r') as file:
            for line in file.readlines():
                curLine = line.strip()
                if curLine == '':
                    continue
                self.data_list.append(curLine)
        del self.data_list[0]  # 删除标题行

        # 划分数据
        if prob != None:
            begin = int(len(self.data_list) * prob[0])
            end = int(len(self.data_list) * prob[1])
            self.data_list = self.data_list[begin:end]

        # 从data_list获取全部X, Y
        for line in self.data_list:
            data = self.get_data(line)
            self.data_map.update({line: data})

        # 随机打乱data_list
        if self.shuffle:
            np.random.shuffle(self.data_list)

    def __len__(self):
        return int(np.ceil(len(self.data_list) / float(self.batch_size)))

    def on_epoch_end(self):
        # 在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle:
            np.random.shuffle(self.data_list)

    def __getitem__(self, idx):
        batch = self.data_list[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = []
        batch_y = []
        for line in batch:
            # 读取出x, y
            [x, y] = self.data_map[line]
            batch_x.append(x)
            batch_y.append(y)

        return np.array(batch_x), np.array(batch_y)

    # 取出X， Y
    def get_data(self, line):
        # 根据文本文件的一行，获得图片名、标签
        imgname, label = line.split(',')

        # 取出一个图片x
        path = os.path.join(self.data_dir, imgname)
        x = imread(filename=path)

        # 数据增强
        if self.data_aug: x = self.aug_pipe.augment_image(x)

        # 预处理并归一化
        x = utils.img_procrss(x)

        # resize
        x = resize(x, dsize=self.img_shape)

        y = utils.process_label(label)
        return x, y

    # 数据增强器
    def augmenter(self):
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        aug = iaa.Sequential(
            [
                iaa.SomeOf((1, 3), [
                    # iaa.PiecewiseAffine(scale=(0.02, 0.03)),
                    iaa.Add(value=(-40, 40), per_channel=0.5),
                    iaa.AdditiveGaussianNoise(scale=(0, 0.1 * 255), per_channel=True),
                    iaa.Multiply((0.5, 1.5), per_channel=0.5)
                ], random_order=True)
            ],
            random_order=True
        )
        return aug

if __name__ == '__main__':
    basepath = r'F:\Code\DL\CAPTCHA'
    data_file = os.path.join(basepath, 'data', 'train', 'train_label.csv')
    data_dir = os.path.join(basepath, 'data', 'train')
    batch_size = 16

    data_gen = DataGenerator(data_file=data_file, data_dir=data_dir, img_shape=(120, 40), batch_size=batch_size,
                             prob=[0.0, 0.2],
                             shuffle=False)
    print(data_gen.data_list.__len__())
    print(data_gen.data_list[0:10])

    X, Y = data_gen.__getitem__(1)
    print(X.shape)
    print(Y.shape)

    print(Y[0])
    print(X[0][0][0])
    imshow(winname='test', mat=X[0])
    waitKey(0)
