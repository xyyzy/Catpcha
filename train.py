import json
import os
import keras.backend.tensorflow_backend as K

from keras.metrics import categorical_accuracy
from util.utils import my_acc
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from keras.optimizers import *
from DataGenerator import DataGenerator
from backend import NerualNetworkModel


# 读取配置文件
config_path = r'.\config.json'
with open(config_path) as config_buffer:
    config = json.loads(config_buffer.read())

model_data = config['model']['model_data']
model_path = config['model']['model_path']

pretrained_weights = config['train']['pretrained_weights']
train_file = config['train']['train_data_file']
train_dir = config['train']['train_data_folder']
train_prob = config['train']['train_prob']

valid_file = config['valid']['valid_data_file']
valid_dir = config['valid']['valid_data_folder']
valid_prob = config['valid']['valid_prob']


def main(load_best_weight=False, load_pre_weight=False):
    model = NerualNetworkModel().LeNet(input_size=(40, 120, 3), regularizer=0.0001, droprate=0.5)
    batch_size = 16

    # 读取预训练权重
    if load_pre_weight:
        try:
            pretrained_weights = r"./model_data/model.118-0.06.h5"
            print(pretrained_weights)
            model.load_weights(filepath=pretrained_weights, by_name=True, skip_mismatch=True)
        except:
            print('pre weights not load')
        else:
            print('pre weights has load')
    #  最好的权重
    elif load_best_weight:
        try:
            print(model_path)
            model.load_weights(filepath=model_path, by_name=True, skip_mismatch=True)
        except:
            print('best weights not load')
        else:
            print('best weights has load')

    plot_model(model=model, to_file=os.path.join(model_data, 'model.png'), show_shapes=True)
    # model.summary()

    # 编译模型
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(0.001, amsgrad=True),
                  metrics=['accuracy', my_acc])

    # 训练数据生成器
    train_data_gen = DataGenerator(data_file=train_file, data_dir=train_dir, img_shape=(120, 40), batch_size=batch_size,
                                   data_aug=True,
                                   prob=train_prob,
                                   shuffle=True)

    # 验证数据生成器
    valid_data_gen = DataGenerator(data_file=valid_file, data_dir=valid_dir, img_shape=(120, 40), batch_size=batch_size,
                                   data_aug=False,
                                   prob=valid_prob,
                                   shuffle=True)

    # 训练
    callbacks = [EarlyStopping(monitor='val_loss', patience=10),
                 CSVLogger(os.path.join(model_data, 'train_log.csv')),
                 ModelCheckpoint(filepath="./model_data/model.{epoch:02d}-{val_loss:.2f}.h5",
                                 save_best_only=False,
                                 period=2)]

    model.fit_generator(train_data_gen, epochs=200, validation_data=valid_data_gen, workers=1, use_multiprocessing=True,
                        callbacks=callbacks)


if __name__ == '__main__':
    main(load_best_weight=False, load_pre_weight=True)
