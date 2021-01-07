# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 18:32:59 2019

@author: qmzhang
"""

import json
import os

import pandas as pd
from cv2 import resize, imread
from keras import Model
from keras.models import load_model

from util import utils
from util.utils import *

config_path = r'./config.json'
with open(config_path) as config_buffer:
    config = json.loads(config_buffer.read())

model_data = config['model']['model_data']
model_path = config['model']['model_path']
data_file = config['predict']['predict_data_file']
data_dir = config['predict']['predict_data_folder']
img_shape = (120, 40)


def model(testpath):
    # your model goes here
    # 在这里放入或者读入模型文件
    model_path = r"./model_data/model.138-0.06.h5" # 0.9774
    model_path = r"./model_data/cnn_best.h5"

    model: Model = load_model(model_path, custom_objects={"my_acc": my_acc})

    X = load_data()
    predict = model.predict(X, batch_size=16)
    ans = utils.decode_predict(predict)
    pass

    # the format of result-file
    # 这里可以生成结果文件
    ids = [str(x) + ".jpg" for x in range(1, 5001)]
    labels = ans
    df = pd.DataFrame([ids, labels]).T
    df.columns = ['ID', 'label']
    return df


def load_data():
    data_list = []
    with open(data_file, 'r') as file:
        for line in file.readlines():
            curLine = line.strip()
            if curLine == '':
                continue
            data_list.append(curLine)
    del data_list[0]  # 删除标题行

    data = []
    for line in data_list:
        data.append(get_data(line))
    return np.array(data)


def get_data(line):
    # 根据文本文件的一行，获得图片名
    imgname = line.split(',')[0]

    # 取出一个x, 并resize
    path = os.path.join(data_dir, imgname)
    x = imread(filename=path)

    # 去噪,并归一化
    x = utils.img_procrss(x)

    x = resize(x, dsize=img_shape)
    return x
