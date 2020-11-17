import os
import sys
import glob
import numpy as np
from skimage import io
#from sklearn import datasets
from sklearn import utils
import cv2
from PIL import Image
 
IMAGE_SIZE_X = 120
IMAGE_SIZE_Y = 80
#COLOR_BYTE = 3
COLOR_BYTE = 1
CATEGORY_NUM = 6
threshhold = 80
 
## ラベル名(0～)を付けたディレクトリに分類されたイメージファイルを読み込む
## 入力パスはラベル名の上位のディレクトリ
def load_png_bin(path):
    # ファイル一覧を取得
    files1 = glob.glob(os.path.join(path, '*/*.png'))
    files = []
    for i in files1:
        #大きいファイルを開いて
        gazo1 = Image.open(i)
        #resizeして
        gazo2 = gazo1.resize((IMAGE_SIZE_X,IMAGE_SIZE_Y))
        #保存
        gazo2.save(i)
        #filesに追加
        files.append(i)

    # イメージとラベル領域を確保
    #images = np.ndarray((len(files), IMAGE_SIZE, IMAGE_SIZE,
    #                      COLOR_BYTE), dtype = np.uint8)
    images = np.ndarray((len(files), IMAGE_SIZE_X, IMAGE_SIZE_Y)
                       , dtype = np.uint8)

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