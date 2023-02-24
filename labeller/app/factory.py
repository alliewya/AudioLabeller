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

    def set_label(self,label):
        self.labels = [label for samp in self.samples]

    def combine_dataset(self, dataset):
        dataset2 = self
        dataset2.path = ""
        dataset2.labels.extend(dataset.labels)
        dataset2.samples.extend(dataset.samples)
        return dataset2

    def k_fold(self, n_k = 5):
        self.crossfolds = KFold(n_splits=n_k, shuffle=True, random_state=5)

    def getMFCC(self, n_mfcc=20):
        self.features = np.empty()
        for samp in self.samples:
            np.append(self.features, librosa.feature.mfcc(
                y=samp.audiodata, sr=samp.samplerate, n_mfcc=n_mfcc), axis=0)
            print(self.features.shape())
    
    def create_frames(self, frame_size, hop_length):
        self.frames = []
    
    def fix_length(self, length):
        samples_new = []
        for audio in self.samples:
            t_len = float(length)*int(self.samplerate)
            audio.audiodata = librosa.util.fix_length(audio.audiodata, size=int(t_len))
            samples_new.append(audio)
        self.samples = samples_new


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

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        if(self.kwargs.get('n_mfcc')):
            self.kwargs


    def features_from_dataset2(self,dataset, **kwargs):
        sr = 22500
        framelength = sr // 2
        hop_lenght = sr // 4
        
        frames = librosa.util.frame(dataset.samples[0], frame_length=framelength, hop_length=hop_lenght)
        features = np.empty


    def features_from_dataset(self, dataset,**kwargs):


        #FRAMES!!!


        n_mfcc = int(self.kwargs.get('n_mfcc',20))
        features = []
        #features = np.empty((0, n_mfcc))
        if(kwargs.get('delta')==True):

            #fix delta!!

            for audio in dataset.samples:
                mfcc = librosa.feature.mfcc(
                    y=audio.audiodata, sr=audio.samplerate, n_mfcc=n_mfcc
                )
                mfcc_delta = librosa.feature.delta(mfcc)
                if(self.kwargs.get('delta2')==True):
                    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
                    mfcc_delta = np.concatenate((mfcc_delta, mfcc_delta2), axis=1)
                mfcc = np.concatenate((mfcc, mfcc_delta), axis=1)
                features = np.concatenate((features, mfcc), axis=0)
        else:    
            for audio in dataset.samples:
                mfcc = librosa.feature.mfcc(
                    y=audio.audiodata, sr=audio.samplerate, n_mfcc=n_mfcc
                )
                mfcc = mfcc.ravel()
                features.append(mfcc)   
        return features

    def single_features(self, audiosample, **kwargs):
        n_mfcc = int(self.kwargs.get('n_mfcc',20))
        features = np.empty((0, n_mfcc))
        np.append(features, librosa.feature.mfcc(
                y=audiosample.audiodata, sr=audiosample.samplerate, n_mfcc=n_mfcc
            ), axis=0)
        return features

    def single_features_from_audio(self, audiofilepath, **kwargs):
        audiosample = AudioSample(path=audiofilepath, samplerate=self.kwargs.get('samplerate',22500))
        features = np.empty((0, kwargs.get('n_mfcc', 20)))
        np.append(features, librosa.feature.mfcc(
                y=audiosample.audiodata, sr=audiosample.samplerate, n_mfcc=n_mfcc
            ), axis=0)
        return features

class MelSpectrogramFeatures(Featuresbase):

    def __init__(self, **kwargs):
        self.kwargs = kwargs


    def features_from_dataset(self, dataset,**kwargs):
        #features = np.empty((0,128))
        features = []
        n_fft = int(self.kwargs.get('n_fft', 2048))
        hop_length = int(self.kwargs.get('hop_length',512))
        window=self.kwargs.get('window','hann')
        center=bool(self.kwargs.get('center',True))
        pad_mode=self.kwargs.get('pad_mode','constant')
        power=float(self.kwargs.get('power',2.0))

        for audio in dataset.samples:
            melspec = librosa.feature.melspectrogram(
                y=audio.audiodata, sr=audio.samplerate, n_fft=n_fft,hop_length=hop_length,
                window=window,center=center,pad_mode=pad_mode,
                power=power
            )
            features.append(melspec)
            #features = np.concatenate((features, melspec),axis=0)
        return features

    def single_features(self, audiosample, **kwargs):
        features = np.empty((0,128))
        melspec = librosa.feature.melspectrogram(
                y=audiosample.audiodata, sr=audiosample.samplerate, n_fft=self.kwargs.get('n_fft', 2048),hop_length=self.kwargs.get('hop_length',512),
                window=self.kwargs.get('window','hann'),center=self.kwargs.get('center',True),pad_mode=self.kwargs.get('pad_mode','constant'),
                power=self.kwargs.get('power',2.0)
            )
        features = np.concatenate((features, melspec),axis=0)
        return features

    def single_features_from_audio(self, audiofilepath, **kwargs):
        audiosample = AudioSample(path=audiofilepath, samplerate=kwargs.get('samplerate',22500))
        features = np.empty((0,128))
        melspec = librosa.feature.melspectrogram(
                y=audiosample.audiodata, sr=audiosample.samplerate, n_fft=self.kwargs.get('n_fft', 2048),hop_length=self.kwargs.get('hop_length',512),
                window=self.kwargs.get('window','hann'),center=self.kwargs.get('center',True),pad_mode=self.kwargs.get('pad_mode','constant'),
                power=self.kwargs.get('power',2.0)
            )
        features = np.concatenate((features, melspec),axis=0)
        return features

    def testfeaturekwargs(self, **kwargs):
        print("hoh")
        print(f' Kwargs: {kwargs}' )
        print(kwargs.get('mel_spec_n_fft'))
        print(kwargs.get('mel_spec_window'))


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




#a = MFCCFeatures().features_from_dataset()