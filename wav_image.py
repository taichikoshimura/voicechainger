import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

def save_png(filename,soundpath,savepath):
    # オーディオファイル(au）を読み込む
    music, fs = librosa.audio.load(soundpath + filename, offset=0.0, duration=7.0)
    # メルスペクトラム（MFCC）変換
    mfccs = librosa.feature.mfcc(music, sr=fs)
    mfccspw = librosa.power_to_db(mfccs, ref=np.max)
    
    # グラフに変換する
    librosa.display.specshow(mfccspw, sr=fs, x_axis='time',cmap="gray")
    # PNG形式画像で保存する
    plt.savefig(savepath + filename + '.png',dpi=200)
    

soundpath = './dataset/1/'
savepath = './save_wav_image/1/'
cnt = 0
for filename in os.listdir(soundpath):
    cnt += 1
    if((cnt % 10) == 0):
        print(cnt,'件を処理しました')
    save_png(filename,soundpath,savepath)