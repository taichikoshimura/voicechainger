import os
import sys
import glob
import numpy as np
import librosa
from skimage import io
#from sklearn import datasets
from sklearn import utils
import cv2

#COLOR_BYTE = 3
COLOR_BYTE = 1
CATEGORY_NUM = 6
threshhold = 80



#ラベル名(0～)を付けたディレクトリに分類されたイメージファイルを読み込む
#入力パスはラベル名の上位のディレクトリ
def load_wavimage_bin(path): 
#     # ファイル一覧を取得
#     files = glob.glob(os.path.join(path, '*/*.wav'))
#     # イメージとラベル領域を確保
#     images = np.ndarray((len(files), 128, 701)
#                            , dtype = np.uint8)
#     labels = np.ndarray(len(files), dtype=np.int) 
#     # イメージとラベルを読み込み
#     for idx, file in enumerate(files):
#         # イメージ読み込み
#         y,sr = librosa.load(path,sr=16000,offset=0.0,duration=7.0)
#         image = librosa.feature.melspectrogram(y=y,
#                                                 sr=sr,
#                                                 n_mels=128,
#                                                 n_fft=512,
#                                                 win_length=480,
#                                                 hop_length=160,)
#        # ディレクトリ名よりラベルを取得
#         label = os.path.split(os.path.dirname(file))[-1]
#         labels[idx] = int(label)
#      # scikit-learn の他のデータセットの形式に合わせる
#     flat_data = images.reshape((-1, 128 * 701 * COLOR_BYTE))
#     images = flat_data.view()
#     #return datasets.base.Bunch(data=images,
#     return utils.Bunch(data=images,
#                  target=labels.astype(np.int),
#                  target_names=np.arange(CATEGORY_NUM),
#                  images=images,
#                  DESCR=None)
    files = glob.glob(os.path.join(path, '*/*.wav'))
    images = np.ndarray((len(files), 128, 302), dtype = np.uint8)
    labels = np.ndarray(len(files), dtype=np.int)
    for idx, file in enumerate(files):
        y,sr = librosa.load(file,offset=0.0,duration=7.0)
        S = librosa.feature.melspectrogram(y=y,sr=sr,n_mels=128)
        image = librosa.amplitude_to_db(S, ref=np.max)

        # ディレクトリ名よりラベルを取得
        label = os.path.split(os.path.dirname(file))[-1]
        labels[idx] = int(label)
        
    flat_data = images.reshape((-1, 128 * 302 * COLOR_BYTE))
    images = flat_data.view()
    return utils.Bunch(data=images,
                 target=labels.astype(np.int),
                 target_names=np.arange(CATEGORY_NUM),
                 images=images,
                 DESCR=None)