import os
import glob
import librosa
from tqdm import tqdm
import numpy as np
from python_speech_features import mfcc, fbank, logfbank
from scipy.sparse import dok_matrix

song = 'audio/a1.m4a'
y, sr = librosa.load(song, sr=16000)
# print(type(y), type(sr))
# print(y.shape, sr)

def extract_features(y, sr=16000, nfilt=10, winsteps=0.02):
    try:
        feat = mfcc(y, sr, nfilt=nfilt, winstep=winsteps)
        return feat
    except:
        raise Exception("Extraction feature error")

def crop_feature(feat, i = 0, nb_step=10, maxlen=100):
    crop_feat = np.array(feat[i : i + nb_step]).flatten()
    print(crop_feat.shape)
    crop_feat = np.pad(crop_feat, (0, maxlen - len(crop_feat)), mode='constant')
    return crop_feat

y, sr = librosa.load(song, sr=16000)
feat = extract_features(y)
print(crop_feature(feat).shape)

data_dir = './audio'
features = []
songs = []

for song in tqdm(os.listdir(data_dir)):
    song = os.path.join(data_dir, song)
    y, sr = librosa.load(song, sr=16000)
    feat = extract_features(y)
    for i in range(0, feat.shape[0] - 10, 5):
        features = features.append(crop_feature(feat, i, nb_step=10))
        songs = songs.append(song)

import pickle

pickle.dump(features, open('features.pk', 'wb'))

pickle.dump(songs, open('songs.pk', 'wb'))

from annoy import AnnoyIndex

f = 100
t = AnnoyIndex(f)

for i in range(len(features)):
    v = features[i]
    t.add_item(i, v)

t.build(100) # 100 trees
t.save('music.ann')

u = AnnoyIndex(f)

u.load('music.ann')

song = os.path.join(data_dir, 'a1.m4a')
y, sr = read_song_frequency(song)
feat = extract_features(y)

results = []
for i in range(0, feat.shape[0], 10):
    crop_feat = crop_feature(feat, i, nb_step=10)
    result = u.get_nns_by_vector(crop_feat, n=5)
    result_songs = [songs[k] for k in result]
    results.append(result_songs)
    
results = np.array(results).flatten()

from collections import Counter

most_song = Counter(results)
most_song.most_common()

# audio_path = '../T08-violin.wav'
# x , sr = librosa.load(audio_path)
# print(type(x), type(sr))
# <class 'numpy.ndarray'> <class 'int'>
# print(x.shape, sr)
# (396688,) 22050