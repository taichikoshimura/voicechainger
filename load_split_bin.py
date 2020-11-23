import os
import sys
import glob
import numpy as np
from skimage import io
#from sklearn import datasets
from sklearn import utils
import cv2
from PIL import Image
 
IMAGE_SIZE_X = 1000
IMAGE_SIZE_Y = 1025
COLOR_BYTE = 3
#COLOR_BYTE = 1
CATEGORY_NUM = 6
threshhold = 80
 
## ラベル名(0～)を付けたディレクトリに分類されたイメージファイルを読み込む
## 入力パスはラベル名の上位のディレクトリ
def load_split_bin(path):
    # ファイル一覧を取得
    files = glob.glob(os.path.join(path, '*/*.png'))

    # イメージとラベル領域を確保
    images = np.ndarray((len(files), IMAGE_SIZE_X, IMAGE_SIZE_Y,
                          COLOR_BYTE), dtype = np.uint8)
    #images = np.ndarray((len(files), IMAGE_SIZE_X, IMAGE_SIZE_Y)
    #                   , dtype = np.uint8)

    labels = np.ndarray(len(files), dtype=np.int)

    # イメージとラベルを読み込み
    for idx, file in enumerate(files):
        # イメージ読み込み
        img = io.imread(file)
        # ディレクトリ名よりラベルを取得
        label = os.path.split(os.path.dirname(file))[-1]
        labels[idx] = int(label)

        # scikit-learn の他のデータセットの形式に合わせる
    flat_data = images.reshape((-1, IMAGE_SIZE_X * IMAGE_SIZE_Y * COLOR_BYTE))
    images = flat_data.view()
    return utils.Bunch(data=images,
                 target=labels.astype(np.int),
                 target_names=np.arange(CATEGORY_NUM),
                 images=images,
                 DESCR=None)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    d = load_handimage_bin("./data/m01")
    plt.figure(figsize=(8, 8))
    # 画像を 2 行 3 列に表示
    k = 0
    for i in range(6):
        for j in range(4):
            img = d.images[d.target==i][j]
            k += 1
            plt.subplot(6,4, k)
            plt.axis('off')
            plt.imshow(img.reshape(40,40), cmap="gray", interpolation='nearest')
            plt.title(i)
    plt.tight_layout()
    plt.show()