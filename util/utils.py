import string

import cv2
import numpy as np
import keras.backend as K
from keras.metrics import categorical_accuracy

wordlist = list('0123456789') + list(string.ascii_letters)


# 处理标签
def process_label(label):
    pro_lab = np.zeros(shape=(4, 62))

    label = list(label)
    for i in range(4):
        c = label[i]
        pro_lab[i][wordlist.index(c)] = 1
    return pro_lab


def decode(arr):
    '''
    :param arr: 一个shape为(4， 64)的二维数组
    :return:    该数组对应的字符串   len = 4
    '''
    arr = arr.tolist()
    result = ''

    idx = list(np.argmax(arr, axis=1))

    for i in idx:
        result += wordlist[i]

    return result


def decode_predict(predict):
    '''
    :param predict: 一个list(ndarray) 其中ndarray.shape = (4, 64), 或者一个shape = (None, 4， 64)的三维数组
    :return:        解码后的字符串列表 list(str) str.len = 4
    '''
    ans = []
    for arr in predict:
        ans.append(decode(arr))
    return ans


def RGBAlgorithm(rgb_img, value=0.5, basedOnCurrentValue=True):
    img = rgb_img * 1.0
    img_out = img

    # 基于当前RGB进行调整（RGB*alpha）
    if basedOnCurrentValue:
        # 增量大于0，指数调整
        if value >= 0:
            alpha = 1 - value
            alpha = 1 / alpha

        # 增量小于0，线性调整
        else:
            alpha = value + 1

        img_out[:, :, 0] = img[:, :, 0] * alpha
        img_out[:, :, 1] = img[:, :, 1] * alpha
        img_out[:, :, 2] = img[:, :, 2] * alpha

    # 独立于当前RGB进行调整（RGB+alpha*255）
    else:
        alpha = value
        img_out[:, :, 0] = img[:, :, 0] + 255.0 * alpha
        img_out[:, :, 1] = img[:, :, 1] + 255.0 * alpha
        img_out[:, :, 2] = img[:, :, 2] + 255.0 * alpha

    img_out = img_out / 255.0

    # RGB颜色上下限处理(小于0取0，大于1取1)
    mask_3 = img_out < 0
    mask_4 = img_out > 1
    img_out = img_out * (1 - mask_3)
    img_out = img_out * (1 - mask_4) + mask_4

    return img_out


def img_procrss(img):
    # 中值滤波
    medBlur = cv2.medianBlur(img, ksize=3)
    # 亮度
    light = RGBAlgorithm(medBlur, value=0.5)
    # 归一化
    result = np.zeros(img.shape, dtype=np.float32)
    cv2.normalize(light, dst=result, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return result

def my_acc(y_true, y_pred):
    out = categorical_accuracy(y_true=y_true, y_pred=y_pred)
    return K.min(out, axis=-1)

if __name__ == '__main__':
    lab1 = process_label('0aa0')
    lab2 = np.zeros(shape=(4, 62))
    lab2[0][10] = 0.9
    lab2[1][2] = 0.9
    lab2[2][2] = 0.9
    lab2[3][2] = 0.9
    res = decode_predict([lab1, lab2])
    print(res)
