from abc import ABC, abstractmethod
import librosa
#from IPython.display import Audio
#import matplotlib.pyplot as plt
import numpy as np
import os, pickle
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier


# ==================================================
#             Data & Dataset
# ==================================================

class AudioSample:

    def __init__(self, path=None, audiodata=None, samplerate=None, sourcefile=None, segmented=False, sourcesplit=None, label=None, normalize=False):
        self.audiodata = audiodata
        self.samplerate = samplerate
        self.sourcefile = sourcefile
        self.segmented = segmented
        self.sourcesplit = sourcesplit
        self.label = label
        if(normalize == True):
            self.normalize_audio()
        if(path):
            self.load(path=path, samplerate=samplerate)
            self.frames = self.get_frames_list()

    def play(self):
        Audio(data=self.audiodata, rate=self.samplerate)

    def load(self, path, samplerate=22050):
        self.audiodata, self.samplerate = librosa.load(path, sr=samplerate)
        self.sourcefile = os.path.basename(path)


    def preview_split(self, threshold=16):
        splitf = librosa.effects.split(self.audiodata, top_db=threshold)
        print("Threshold set at {} db below max".format(threshold))
        print("Number of segments found: {}".format(splitf.shape[0]))
        fig, ax = plt.subplots(nrows=3, sharex=True)
        fig.tight_layout(pad=1.2)
        librosa.display.waveshow(y, sr=sr, ax=ax[0], x_axis='s')
        ax[0].set(title='Envelope view, mono')
        ax[0].label_outer()

        D = librosa.stft(y)  # STFT of y
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        librosa.display.specshow(S_db, ax=ax[1], x_axis='s', y_axis='hz')
        ax[1].set(title='Spectogram')
        ax[1].label_outer()

        mel_f2 = librosa.feature.melspectrogram(y=y, sr=sr)
        S_db_mel = librosa.amplitude_to_db(np.abs(mel_f2), ref=np.max)

        librosa.display.specshow(S_db_mel, ax=ax[2], y_axis='mel', x_axis='s')
        ax[2].set(title='Mel Spectogram')
        ax[2].label_outer()

        for i in range(splita.shape[0]):
            ax[0].axvspan((splita[i][0]/sr), (splita[i][1]/sr),
                          color="green", alpha=0.3)
            ax[1].axvspan((splita[i][0]/sr), (splita[i][1]/sr),
                          color="green", alpha=0.3)
            ax[2].axvspan((splita[i][0]/sr), (splita[i][1]/sr),
                          color="green", alpha=0.3)

    def get_length(self):
        return(librosa.get_duration(self.audiodata, self.samplerate))

    def get_segments(self, threshold=16):
        splitpoints = librosa.effects.split(self.audiodata, top_db=threshold)
        segs = []
        for i in range(splitpoints.shape[0]):
            segs.append(AudioSample(audiodata=(self.audiodata[splitpoints[i][0]:splitpoints[i][1]]), samplerate=self.samplerate,
                        sourcefile=self.sourcefile, segmented=True, sourcesplit=splitpoints[i], label=self.label))
        return segs

    def get_frames_list(self, framelength=11250, hop_length=5625):
        a = self.audiodata
        targetlen = max(framelength, (((len(a) // hop_length)+1)*hop_length))
        a = librosa.util.fix_length(a, size=targetlen)
        frames = librosa.util.frame(
            a, frame_length=framelength, hop_length=hop_length)
        self.frames = [frame for frame in frames.T]
        return self.frames

    def normalize_audio(self):
        a = self.audiodata
        a = np.nan_to_num(a)
        a = librosa.util.normalize(a, norm=1)
        self.audiodata = a
        return self.audiodata


class Dataset:

    def __init__(self, path1=None, samplerate=None, labels=None, load=False, config = None):
        self.path = path1
        self.samplerate = samplerate
        self.labels = labels
        self.samples = []
        self.segmented = False
        self.features = []
        self.crossfolds = False
        if(load):
            filelist = librosa.util.find_files(self.path, recurse=False)
            print(str(len(filelist))+" Files to load")
            remain = len(filelist)
            count = 0
            for f in filelist:
                file1 = os.path.join(self.path, f)
                samp = AudioSample(
                    path=file1, samplerate=self.samplerate, sourcefile=f)
                self.samples.append(samp)
                count = count + 1
                if(count % 20 == 0):
                    print(remain-count)
        if(config["pickled"]["enabled"]==True):
            with open(config["pickeld"]["filename"], "rb") as f:
                datasetemp = pickle.load(f)
                self.samplerate = datasetemp.samplerate
                self.labels = datasetemp.labels
                self.samples = datasetemp.samples
                self.segmented = datasetemp.segmented
                self.features = datasetemp.features
                

    def segment(self, threshold=16):
        segmentedlist = []
        for sample in self.samples:
            sampsegs = sample.get_segments(threshold=threshold)
            for x in sampsegs:
                segmentedlist.append(x)
        print("Created {} segmented audio samples".format(len(segmentedlist)))
        self.samples = segmentedlist
        self.segmented = True

    def labelsasfilenames(self):
        self.labels = []
        for samp in self.samples:
            samp.label = samp.sourcefile
            self.labels.append(samp.sourcefile)

    def k_fold(self, n_k = 5):
        self.crossfolds = KFold(n_splits=n_k, shuffle=True, random_state=5)

    def getMFCC(self, n_mfcc=20):
        self.features = np.empty()
        for samp in self.samples:
            np.append(self.features, librosa.feature.mfcc(
                y=samp.audiodata, sr=samp.samplerate, n_mfcc=n_mfcc), axis=0)
            print(self.features.shape())

# =======================================================
#               Features
# =======================================================

class Featuresbase():
    """An abstract base class for all feature extractors"""

    @abstractmethod
    def features_from_dataset():
        pass

    @abstractmethod
    def single_features():
        pass

    @abstractmethod
    def single_features_from_audio(audiofilepath):
        pass

class MFCCFeatures(Featuresbase):

    def features_from_dataset(dataset,**kwargs):
        features = np.empty((0, kwargs.get('n_mfcc', 20)))
        if(kwargs.get('delta')==True):
            for audio in dataset.samples:
                mfcc = librosa.feature.mfcc(
                    y=audio.audiodata, sr=audio.samplerate, n_mfcc=kwargs.get('n_mfcc', 20)
                )
                mfcc_delta = librosa.feature.delta(mfcc)
                if(kwargs.get('delta2')==True):
                    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
                    mfcc_delta = np.concatenate((mfcc_delta, mfcc_delta2), axis=1)
                mfcc = np.concatenate((mfcc, mfcc_delta), axis=1)
                features = np.concatenate((features, mfcc), axis=0)
        else:    
            for audio in dataset.samples:
                mfcc = librosa.feature.mfcc(
                    y=audio.audiodata, sr=audio.samplerate, n_mfcc=kwargs.get('n_mfcc', 20)
                )
                features = np.concatenate((features, mfcc), axis=0)
            
        return features

    def single_features(audiosample, **kwargs):
        features = np.empty((0, kwargs.get('n_mfcc', 20)))
        np.append(features, librosa.feature.mfcc(
                y=audiosample.audiodata, sr=audiosample.samplerate, n_mfcc=kwargs.get('n_mfcc', 20)
            ), axis=0)
        return features

    def single_features_from_audio(audiofilepath, **kwargs):
        audiosample = AudioSample(path=audiofilepath, samplerate=kwargs.get('samplerate',22500))
        features = np.empty((0, kwargs.get('n_mfcc', 20)))
        np.append(features, librosa.feature.mfcc(
                y=audiosample.audiodata, sr=audiosample.samplerate, n_mfcc=kwargs.get('n_mfcc', 20)
            ), axis=0)
        return features

class MelSpectrogramFeatures(Featuresbase):

    def features_from_dataset(dataset,**kwargs):
        features = np.empty((0,128))
        for audio in dataset.samples:
            melspec = librosa.feature.melspectrogram(
                y=audio.audiodata, sr=audio.samplerate, n_fft=kwargs.get('n_fft', 2048),hop_length=kwargs.get('hop_length',512),
                window=kwargs.get('window','hann'),center=kwargs.get('center',True),pad_mode=kwargs.get('pad_mode','constant'),
                power=kwargs.get('power',2.0)
            )
            features = np.concatenate((features, melspec),axis=0)
        return features

    def single_features(audiosample, **kwargs):
        features = np.empty((0,128))
        melspec = librosa.feature.melspectrogram(
                y=audiosample.audiodata, sr=audiosample.samplerate, n_fft=kwargs.get('n_fft', 2048),hop_length=kwargs.get('hop_length',512),
                window=kwargs.get('window','hann'),center=kwargs.get('center',True),pad_mode=kwargs.get('pad_mode','constant'),
                power=kwargs.get('power',2.0)
            )
        features = np.concatenate((features, melspec),axis=0)
        return features

    def single_features_from_audio(audiofilepath, **kwargs):
        audiosample = AudioSample(path=audiofilepath, samplerate=kwargs.get('samplerate',22500))
        features = np.empty((0,128))
        melspec = librosa.feature.melspectrogram(
                y=audiosample.audiodata, sr=audiosample.samplerate, n_fft=kwargs.get('n_fft', 2048),hop_length=kwargs.get('hop_length',512),
                window=kwargs.get('window','hann'),center=kwargs.get('center',True),pad_mode=kwargs.get('pad_mode','constant'),
                power=kwargs.get('power',2.0)
            )
        features = np.concatenate((features, melspec),axis=0)
        return features

class FeaturesFactory():
    """Features class that returns a feature extractor"""

    def __init__(self, **kwargs):
        pass



# =======================================================
#               Model Section
# =======================================================

class Modelbase():
    """An abstract base class for all pipeline models"""

    @abstractmethod
    def fit(self, X, y):
        """"""
        pass

    @abstractmethod
    def predict(self, X, y):
        """"""
        pass

    @abstractmethod
    def evaluate(self, X, y):
        """"""
        pass

class KnnModel(Modelbase):

    def __init__(self, **kwargs):
        self.model = KNeighborsClassifier(**kwargs, n_jobs=-1)

    def fit(self, X, y):
        self.model.fit(X,y)
    
    def predict(self, X, y):
        return self.model.predict(X)

    def evaluate(self, X, y):
        return self.model.score(X,y)


class ModelFactory:
    """Factory class that returns a model"""

    def create_model(model_type, **kwargs):
        if model_type == "knn":
            return KnnModel(**kwargs)




a = MFCCFeatures().features_from_dataset()