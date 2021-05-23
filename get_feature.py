# import keras
from sklearn.utils import shuffle
from tqdm import tqdm
import librosa
import librosa.display
import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
root_path = "F:/dataset/pig_voice_origion"
train = os.path.join(root_path, "train")
val = os.path.join(root_path, "val")
test = os.path.join(root_path, "test")


def get_features(path, f='MC'):
    y, sr = librosa.load(path)
    hop_l = 512 * 2
    f1 = librosa.feature.mfcc(y, sr, n_mfcc=60, hop_length=hop_l)
    f2 = librosa.feature.chroma_stft(y, sr, hop_length=hop_l)
    f3 = librosa.feature.spectral_contrast(y, sr, hop_length=hop_l)
    f4 = librosa.effects.harmonic(y)
    f4 = librosa.feature.tonnetz(f4, sr, hop_length=hop_l)
    f5 = librosa.feature.melspectrogram(y, sr, n_mels=60, hop_length=hop_l)
    f5 = librosa.power_to_db(f5)
    LMC = np.vstack((f1, f2, f3, f4))
    MC = np.vstack((f5, f2, f3, f4))
    MLMC = np.vstack((f1, f5, f2, f3, f4))
    features = {"f1": f1,"f2":f2,"f3":f3,"f4":f4,"f5":f5,"LMC": LMC, "MC": MC, "MLMC": MLMC}
    return features[f]


def get_data(path, f='MLMC'):
    classnumber = [0, 0, 0, 0]
    path_list = []
    label_list = []
    paths = os.listdir(path)
    paths = shuffle(paths, random_state=21)
    for i in paths:
        path_list.append(os.path.join(path, i))
        label = int(i.split("-")[1].split(".")[0])
        label_list.append(label)
        classnumber[label] += 1

    datas = []
    for i in tqdm(path_list):
        datas.append(get_features(i,f))
    wav_max = 55

    for i in range(len(datas)):
        while (datas[i].shape[1] < wav_max):
            datas[i] = np.c_[datas[i], np.zeros(datas[i].shape[0])]

    return datas, label_list, classnumber


f = 'MC'
train_data, train_labels, train_classes = get_data(train, f=f)
print(np.array(train_data).shape, np.array(train_labels).shape, train_classes)

val_data, val_labels, val_classes = get_data(val, f=f)
print(np.array(val_data).shape, np.array(val_labels).shape, val_classes)

test_data, test_labels, test_classes = get_data(test, f=f)
print(np.array(test_data).shape, np.array(test_labels).shape, test_classes)

np.save("./"+f+"/"+"train_" + f + ".npy", np.array(train_data))
np.save("./"+f+"/"+"train_label.npy", np.array(train_labels))

np.save("./"+f+"/"+"val_" + f + ".npy", np.array(val_data))
np.save("./"+f+"/"+"val_label.npy", np.array(val_labels))

np.save("./"+f+"/"+"test_" + f + ".npy", np.array(test_data))
np.save("./"+f+"/"+"test_label.npy", np.array(test_labels))
