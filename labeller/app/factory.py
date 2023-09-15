from abc import ABC, abstractmethod
import librosa
import librosa.display
#from IPython.display import Audio
import matplotlib.pyplot as plt
import numpy as np
import os, pickle, time
from datetime import datetime
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import traceback, math

import multiprocessing
from hmmlearn import hmm


from scipy.stats import skew, kurtosis

#tempforpi
import scipy.signal as signal
import scipy.fftpack as fft
from scipy.fft import dct


#Discrete wavelet
import pywt
import soundfile as sf


#
from kymatio.numpy import Scattering1D


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

    def get_segments(self, threshold=60):
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

    def get_frames_as_audioobj(self, framelength=11250, hop_length=5625):
        a = self.audiodata
        targetlen = max(framelength, (((len(a) // hop_length)+1)*hop_length))
        a = librosa.util.fix_length(a, size=targetlen)
        frames = librosa.util.frame(
            a, frame_length=framelength, hop_length=hop_length)
        self.frames = [frame for frame in frames.T]
        frames_list = []
        for frame in self.frames:
            frames_list.append(AudioSample(audiodata=frame,label=self.label, samplerate=self.samplerate))
        return frames_list

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
        self.descriptors = None
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

                

    def segment(self, threshold=60):
        segmentedlist = []
        for sample in self.samples:
            sampsegs = sample.get_segments(threshold=threshold)
            for x in sampsegs:
                segmentedlist.append(x)
        print("Unsegmented audio samples".format(len(self.samples)))        
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
        for samp in self.samples:
            samp.label = label

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
        print("Creating Frames")
        print("Samples Length" + str(len(self.samples)))
        print("Labels length" + str(len(self.labels)))
        samples_new = []
        labels_new = []
        for audio in self.samples:
            #audio.get_frames_list(framelength=frame_size, hop_length=hop_length)
            frames = audio.get_frames_as_audioobj(framelength=frame_size, hop_length=hop_length)
            for x in frames:
                samples_new.append(x)
                labels_new.append(x.label)
        self.samples = samples_new
        self.labels = labels_new
        print("Created Frames")
        print("Frames Length" + str(len(self.samples)))
        print("Labels length" + str(len(self.labels)))
        

    def get_labels(self):
        labels = []
        for samp in self.samples:
            labels.append(samp.label)
        return labels

    def add_labels_to_audiosamps(self):
        for i, audio in enumerate(self.samples):
            audio.label = self.labels[i]
    
    def fix_length(self, length):
        samples_new = []
        for audio in self.samples:
            t_len = float(length)*int(self.samplerate)
            audio.audiodata = librosa.util.fix_length(audio.audiodata, size=int(t_len))
            samples_new.append(audio)
        self.samples = samples_new

    def trim_audio(self, **kwargs):
        print("Pre-trim Average length: " + str(self.get_average_length()))
        for audio in self.samples:
            audio.audiodata, index = librosa.effects.trim(y=audio.audiodata,ref=3)
        print("Post-trim Average length: " + str(self.get_average_length()))


    def get_average_length(self):
        total_length = 0
        for audio in self.samples:
            total_length = total_length + librosa.get_duration(y=audio.audiodata)
        return total_length/len(self.samples)
    
    def get_length_stats(self):
        total_length = 0
        for audio in self.samples:
            total_length = total_length + librosa.get_duration(y=audio.audiodata)
        count = len(self.samples)
        average = total_length/count
        return total_length, average, count

    def normalize(self):
        for audio in self.samples:
            audio.audiodata = librosa.util.normalize(S = audio.audiodata)

    def return_statistics(self):
        stats = {}

        stats["Total Length"], stats["Average Length"], stats["Audio"] = self.get_length_stats()
        stats["Label 1"] = sum(self.labels)
        stats["Label 0"] = len(self.labels)- stats["Label 1"]
        stats["Sample Rate"] = self.samples[0].samplerate
        return stats


# ==================================================
#             Preprocessing
# ==================================================



class Preprocessorbase():
    """An abstract base class for all preprocessors"""
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    @abstractmethod
    def process_sample(self):
        pass

    #Process the dataset using multiple processes
    def preprocess_audio_dataset_multi(self, dataset, **kwargs):
        start_time = time.time()

        with multiprocessing.Pool(processes=12) as pool:
            processed_audio = pool.starmap(
                self.process_sample,
                [(audio, kwargs) for audio in dataset.samples],
                chunksize=250
            )

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Pre-Processing time: {elapsed_time} seconds")

        dataset.samples = processed_audio
        return dataset

    #Print the input config
    def testfeaturekwargs(self, **kwargs):
        print(f' Kwargs: {kwargs}' )


class WaveletDenoise(Preprocessorbase):
    
    def process_sample(self, audio, kwargs):
        
        #sample_rate = 22050
        #folder_path = 'G:/Cough/Wavelet/denoise/alt/quart/level5-5/'
        folder_path = 'G:/Cough/Wavelet/mfcc-norm-trim'
        folder_path2 = 'G:/Cough/Wavelet/amplitude-norm-trim'
        folder_path3 = 'G:/Cough/Wavelet/mfcc-amplitude-norm-trim'

        #extract_save_amplitude(audio.audiodata, (str(audio.label)+"_"+audio.sourcefile[:-4]+"_org"), folder_path2, audio.samplerate)
        #extract_save_mfcc(audio.audiodata, (str(audio.label)+"_"+audio.sourcefile[:-4]+"_org"), folder_path, audio.samplerate)
        #plot_mfcc_and_amplitude(audio.audiodata, (str(audio.label)+"_"+audio.sourcefile[:-4]+"_org"), folder_path3, audio.samplerate)
        
        wavelet = kwargs.get("wavelet","haar")
        level = int(kwargs.get("level",3))
        magnitude = float(kwargs.get("magnitude",2))
        include_level = int(kwargs.get("include_level",(level+1)))
        threshold_enable = bool(kwargs.get("threshold_enable",True))
        #extract_save_melspec(audio.audiodata, (str(audio.label)+"_"+audio.sourcefile[:-4]+"_org"), folder_path, audio.samplerate)
        #sf.write((folder_path+str(audio.label)+"_"+audio.sourcefile[:-4]+"_org"+".wav"), audio.audiodata, audio.samplerate)

        #denoise
        coeffs = pywt.wavedec(audio.audiodata, wavelet, level=level)


        
        #Speed Up Audio!
        coeffs = coeffs[:include_level]
        
        #Remove not included subbands
        # for i in range(len(coeffs)):
        #     if i >= include_level:
        #         #print("thresh"+str(i))
        #         coeffs[i] = np.zeros_like(coeffs[i])

        #alt for all
        #threshold = self.universal_threshold(np.concatenate(coeffs))

        if(threshold_enable):
            # print("Thresh")
            for i in range(len(coeffs)):
                
                threshold = self.universal_threshold(coeffs[i])
                #print(threshold)
                
                #print(coeffs[i])
                if(threshold!=0):
                    #print(float(threshold)*magnitude)
                    #print(type(self.universal_threshold(coeffs[i])))
                    #print(type(magnitude))
                    coeffs[i] = pywt.threshold(coeffs[i],(threshold*magnitude))
                #coeffs[i] = self.sure_threshold(coeffs[i],sigma)
                #coeffs[i] = pywt.threshold(coeffs[i],self.universal_threshold(coeffs[i]))
    

        #coeffs = [pywt.threshold(c, (np.std(c)*2),'hard') for c in coeffs]
        denoised_audio = pywt.waverec(coeffs, wavelet)
        #extract_save_melspec(denoised_audio, (str(audio.label)+"_"+audio.sourcefile[:-4]+"_denoise"), folder_path, audio.samplerate)
        #sf.write((folder_path+str(audio.label)+"_"+audio.sourcefile[:-4]+"_denoise"+".wav"), denoised_audio, audio.samplerate)
        audio.audiodata = denoised_audio

        return audio
    
    def sure_threshold(self, coeffs, sigma):
        num_coeffs = len(coeffs)
        var_noise = sigma**2
        lambda_factor = np.sqrt(2 * np.log(num_coeffs))

        thresholded_coeffs = []
        for i in range(num_coeffs):
            coeff_i = coeffs[i]
            threshold_i = lambda_factor * np.sqrt(var_noise + 2 * np.log(num_coeffs))
            thresholded_i = pywt.threshold(coeff_i, threshold_i, 'soft').tolist()
            thresholded_coeffs.append(thresholded_i)

        return thresholded_coeffs
    
    def universal_threshold(self, data):
        # Computes the universal threshold using the Donoho-Johnstone heuristic, which is based on the data's standard deviation and the signal length.
          # Compute the standard deviation of the data
        sigma = np.std(data)

        if sigma == 0:
                # If sigma is zero, return the original data
                return 0

        # Calculate the universal threshold using the Donoho-Johnstone heuristic
        threshold = sigma * np.sqrt(2 * np.log(len(data)))

        return threshold


    # def process_sample(self, audio, kwargs):
        
    #     #sample_rate = 22050
    #     # extract_save_melspec(audio.audiodata, (audio.sourcefile[:-4]+"_org"), None, audio.samplerate)

    #     wavelet = kwargs.get("wavelet","haar")
    #     level = kwargs.get("level",3)
    #     magnitude = kwargs.get("magnitude",2)

    #     #denoise
    #     coeffs = pywt.wavedec(audio.audiodata, wavelet, level=level)
    #     # Apply thresholding to the wavelet coefficients
    #     threshold = np.std(coeffs[-1]) * 1  # Adjust the threshold as needed - threshold is set as twice the standard deviation of the last level coefficients. 
    #     # print(threshold)
    #     #coeffs = [pywt.threshold(c, threshold,'hard') for c in coeffs]
    #     o = np.median(coeffs[-1])/0/6745

    #     for i in range(len(coeffs)):
    #         threshold = np.std(coeffs[i]) * magnitude
    #         if threshold != 0:
    #             coeffs[i] = pywt.threshold(coeffs[i], threshold, 'soft')


    #     #coeffs = [pywt.threshold(c, (np.std(c)*2),'hard') for c in coeffs]
    #     denoised_audio = pywt.waverec(coeffs, wavelet)
    #     # extract_save_melspec(denoised_audio, (audio.sourcefile[:-4]+"_denoise"), None, audio.samplerate)
    #     audio.audiodata = denoised_audio

    #     return audio
    
    # def process_sample(self, audio, kwargs):
        
    #     #sample_rate = 22050
    #     # extract_save_melspec(audio.audiodata, (audio.sourcefile[:-4]+"_org"), None, audio.samplerate)

    #     wavelet = kwargs.get("wavelet","haar")
    #     level = kwargs.get("level",3)
    #     magnitude = kwargs.get("magnitude",2)

    #     #denoise
    #     coeffs = pywt.wavedec(audio.audiodata, 'db4', level=5)
    #     # Apply thresholding to the wavelet coefficients
    #     threshold = np.std(coeffs[-1]) * 1  # Adjust the threshold as needed - threshold is set as twice the standard deviation of the last level coefficients. 
    #     # print(threshold)
    #     #coeffs = [pywt.threshold(c, threshold,'hard') for c in coeffs]
    #     for c in coeffs:
    #         threshold = np.std(c)*2
    #         if(threshold!=0):
    #             c = pywt.threshold(c, threshold,'soft')


    #     #coeffs = [pywt.threshold(c, (np.std(c)*2),'hard') for c in coeffs]
    #     denoised_audio = pywt.waverec(coeffs, 'db4')
    #     # extract_save_melspec(denoised_audio, (audio.sourcefile[:-4]+"_denoise"), None, audio.samplerate)
    #     audio.audiodata = denoised_audio

    #     return audio


    # def process_sample(self, audio, kwargs):
    #     # Define the thresholding function
    #     def thresholding_function(value, magnitude):
    #         with np.errstate(divide='ignore', invalid='ignore'):
    #             thresholded = np.where(magnitude != 0, (1 - value / magnitude), value)
    #         return thresholded


    #     wavelet = kwargs.get("wavelet","haar")
    #     level = kwargs.get("level",3)
    #     magnitude = kwargs.get("magnitude",2)
        
    #     extract_save_melspec(audio.audiodata, (audio.sourcefile[:-4]+"_org"), None, audio.samplerate)
    #     # Denoise
    #     coeffs = pywt.wavedec(audio.audiodata, wavelet, level=level)
    #     # Apply thresholding to the wavelet coefficients
    #     threshold = np.std(coeffs[-1]) * magnitude
    #     coeffs = [thresholding_function(c, threshold) for c in coeffs]
    #     denoised_audio = pywt.waverec(coeffs, wavelet)
    #     extract_save_melspec(denoised_audio, (audio.sourcefile[:-4]+"_denoise"), None, audio.samplerate)
    #     audio.audiodata = denoised_audio

    #     return audio



# =======================================================
#               Features
# =======================================================

class FeaturesBase():
    """An abstract base class for all feature extractors"""

    def __init__(self, **kwargs):
            self.kwargs = kwargs

    @abstractmethod
    def features_from_dataset():
        pass

    @abstractmethod
    def single_features():
        pass

    @abstractmethod
    def single_features_from_audio(audiofilepath):
        pass

    @abstractmethod
    def process_sample():
        pass

    #Process the dataset using multiple processes
    def features_from_dataset_multi(self, dataset,**kwargs):
        with multiprocessing.Pool(processes=12) as pool:
            results = pool.starmap(
                self.process_sample,
                [(audio, ) for audio in dataset.samples]
            )
        features = [x['feature'] for x in results]
        labels = [x['label'] for x in results]
        return features, labels

    #Print the input config
    def testfeaturekwargs(self, **kwargs):
        print(f' Kwargs: {kwargs}' )


class MFCCFeatures(FeaturesBase):

    def __init__(self, **kwargs):
        self.kwargs = kwargs
    


    def features_from_dataset2(self,dataset, **kwargs):
        sr = 22050
        framelength = sr // 2
        hop_lenght = sr // 4
        
        frames = librosa.util.frame(dataset.samples[0], frame_length=framelength, hop_length=hop_lenght)
        features = np.empty


    def features_from_dataset(self, dataset,**kwargs):

        kwargs = self.kwargs
        n_mfcc = int(kwargs.get('n_mfcc', 20))
        center = bool(kwargs.get('center', False))
        dct_type = int(kwargs.get('dct_type', 2))
        norm = kwargs.get('norm', 'ortho')
        n_fft = int(kwargs.get('n_fft', 2048))
        hop_length = int(kwargs.get('hop_length', 512))

        #FRAMES!!!


        #n_mfcc = int(self.kwargs.get('n_mfcc',20))
        features = []
        labels = []
        templist = []
        #features = np.empty((0, n_mfcc))
        if(bool(kwargs.get('delta'))==True):
            print("Delta")
            #fix delta!!

            for audio in dataset.samples:
                mfcc = librosa.feature.mfcc(
                    y=audio.audiodata, sr=audio.samplerate, **self.kwargs
                )
                mfcc_delta = librosa.feature.delta(mfcc)
                if(bool(self.kwargs.get('delta2'))==True):
                    print("Delta Delta")
                    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
                    mfcc_delta = np.concatenate((mfcc_delta, mfcc_delta2), axis=1)
                mfcc = np.concatenate((mfcc, mfcc_delta), axis=1)
                mfcc = mfcc.ravel()
                #features = np.concatenate((features, mfcc), axis=0)
                templist.append({'feature':mfcc, 'label':audio.label})
        else:    
            for audio in dataset.samples:
                # mfcc = librosa.feature.mfcc(
                #     y=audio.audiodata, sr=audio.samplerate, **self.kwargs
                # )
                # mfcc = librosa.feature.mfcc(
                #     y=audio.audiodata, sr=audio.samplerate, n_mfcc=n_mfcc
                # )
                mfcc = librosa.feature.mfcc(
                    y=audio.audiodata, sr=audio.samplerate, n_mfcc=n_mfcc, center=center, dct_type=dct_type, norm=norm, n_fft=n_fft, hop_length=hop_length
                )
                mfcc = mfcc.ravel()
                templist.append({'feature':mfcc, 'label':audio.label})
                #features.append(mfcc)
        features = [x['feature'] for x in templist]
        labels = [x['label'] for x in templist]
        return features,labels

    def single_features(self, audiosample=None,audioframe = None, **kwargs):
        # n_mfcc = int(self.kwargs.get('n_mfcc',20))
        # features = np.empty((0, n_mfcc))
        # np.append(features, librosa.feature.mfcc(
        #         y=audiosample.audiodata, sr=audiosample.samplerate, n_mfcc=n_mfcc
        #     ), axis=0)


        # kwargs = self.kwargs
        # n_mfcc = int(kwargs.get('n_mfcc', 20))
        # center = bool(kwargs.get('center', False))
        # dct_type = int(kwargs.get('dct_type', 2))
        # norm = kwargs.get('norm', 'ortho')
        # n_fft = int(kwargs.get('n_fft', 2048))
        # hop_length = int(kwargs.get('hop_length', 512))
        # features = []
        # mfcc = librosa.feature.mfcc(
        #     y=audioframe, sr=22050, n_mfcc=n_mfcc, center=center, dct_type=dct_type, norm=norm, n_fft=n_fft, hop_length=hop_length
        # )
        # mfcc = mfcc.ravel()
        # features.append(mfcc)
        kwargs = self.kwargs
        n_mfcc = int(kwargs.get('n_mfcc', 20))
        center = bool(kwargs.get('center', False))
        dct_type = int(kwargs.get('dct_type', 2))
        norm = kwargs.get('norm', 'ortho')
        n_fft = int(kwargs.get('n_fft', 2048))
        hop_length = int(kwargs.get('hop_length', 512))
        mfcc = librosa.feature.mfcc(
                y=audioframe, sr=22050, n_mfcc=n_mfcc, center=center, dct_type=dct_type, norm=norm, n_fft=n_fft, hop_length=hop_length
            )
        if(bool(kwargs.get('delta'))==True):
            mfcc_delta = librosa.feature.delta(mfcc)
            if(bool(self.kwargs.get('delta2'))==True):
                mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
                mfcc_delta = np.concatenate((mfcc_delta, mfcc_delta2), axis=1)
            mfcc = np.concatenate((mfcc, mfcc_delta), axis=1)
        mfcc = mfcc.ravel()
        
        features = np.asarray(mfcc)
        return features

    def single_features_from_audio(self, audiofilepath, **kwargs):
        audiosample = AudioSample(path=audiofilepath, samplerate=self.kwargs.get('samplerate',22050))
        features = np.empty((0, kwargs.get('n_mfcc', 20)))
        np.append(features, librosa.feature.mfcc(
                y=audiosample.audiodata, sr=audiosample.samplerate, n_mfcc=n_mfcc
            ), axis=0)
        features = features.ravel()
        return features

    def process_sample(self, audio):
        kwargs = self.kwargs
        n_mfcc = int(kwargs.get('n_mfcc', 20))
        center = bool(kwargs.get('center', False))
        dct_type = int(kwargs.get('dct_type', 2))
        norm = kwargs.get('norm', 'ortho')
        n_fft = int(kwargs.get('n_fft', 2048))
        hop_length = int(kwargs.get('hop_length', 512))
        n_mfcc_trun = int(kwargs.get('n_mfcc_trun', n_mfcc))
        #print(n_mfcc_trun)

        mfcc = librosa.feature.mfcc(
            y=audio.audiodata, sr=audio.samplerate, n_mfcc=n_mfcc, center=center, dct_type=dct_type,
            norm=norm, n_fft=n_fft, hop_length=hop_length
        )
        
        #print(mfcc.shape)
        if(n_mfcc_trun < n_mfcc):
            mfcc = mfcc.T[:n_mfcc_trun].T
            #print("oo")
            #print(mfcc.shape)
        #mfcc = mfcc[:n_mfcc_trun]  # Take only the first n_mfcc_trun number of MFCCs
        
        if bool(kwargs.get('delta')):
            mfcc_delta = librosa.feature.delta(mfcc)
            if bool(kwargs.get('delta2')):
                mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
                mfcc_delta = np.concatenate((mfcc_delta, mfcc_delta2), axis=1)
            mfcc = np.concatenate((mfcc, mfcc_delta), axis=1)
        
        mfcc = mfcc.ravel()
        
        return {'feature': mfcc, 'label': audio.label}



class MelSpectrogramFeatures(FeaturesBase):

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
        
        audiosample = AudioSample(path=audiofilepath, samplerate=kwargs.get('samplerate',22050))
        features = np.empty((0,128))
        melspec = librosa.feature.melspectrogram(
                y=audiosample.audiodata, sr=audiosample.samplerate, n_fft=self.kwargs.get('n_fft', 2048),hop_length=self.kwargs.get('hop_length',512),
                window=self.kwargs.get('window','hann'),center=self.kwargs.get('center',True),pad_mode=self.kwargs.get('pad_mode','constant'),
                power=self.kwargs.get('power',2.0)
            )
        features = np.concatenate((features, melspec),axis=0)
        return features

    def process_sample(self, audio):
        kwargs = self.kwargs
        melspec = librosa.feature.melspectrogram(
                y=audio.audiodata, sr=audio.samplerate, n_fft=int(self.kwargs.get('n_fft', 2048)),hop_length=int(self.kwargs.get('hop_length',512)),
                window=self.kwargs.get('window','hann'),center=self.kwargs.get('center',True),pad_mode=self.kwargs.get('pad_mode','constant'),
                power=float(self.kwargs.get('power',2.0))
            )
        melspec = melspec.ravel()
        
        return {'feature': melspec, 'label': audio.label}


class WaveletDecFeatures(FeaturesBase):

    def __init__(self, **kwargs):
            self.kwargs = kwargs
        

    def features_from_dataset2(self,dataset, **kwargs):
        sr = 22050
        framelength = sr // 2
        hop_lenght = sr // 4
        
        frames = librosa.util.frame(dataset.samples[0], frame_length=framelength, hop_length=hop_lenght)
        features = np.empty



    def process_sample(self, audio, kwargs):

        sample_rate = 22050

        # Maximum useful decomposition level
        # w = pywt.Wavelet('db4')
        # print (pywt.dwt_max_level(data_len=len(audio.audiodata), filter_len=w.dec_len))
        
        # lvl = int(self.kwargs.get('level',None))
        # cooef = pywt.wavedec(audio.audiodata, wavelet=self.kwargs.get('wavelet', 'haar'), mode=self.kwargs.get('mode', 'sym'), level=lvl)
        # print(len(cooef))
        # print(type(cooef[0]))


        # melspec = librosa.feature.melspectrogram(
        #         y=audio.audiodata, sr=audio.samplerate, n_fft=self.kwargs.get('n_fft', 2048),hop_length=self.kwargs.get('hop_length',512),
        #         window=self.kwargs.get('window','hann'),center=self.kwargs.get('center',True),pad_mode=self.kwargs.get('pad_mode','constant'),
        #         power=self.kwargs.get('power',2.0)
        #     )
        # fn = str(audio.label) + "_" + audio.sourcefile[:-4] + "org"
        # savemelspec([melspec,fn,audio.samplerate])
        
        # wafn = "C:/Users/Alliewya/Documents/Cough Monitor/Spectrograms/Denoise_2/" + fn + ".wav"
        # sf.write(wafn, audio.audiodata, audio.samplerate)


        #denoise
        coeffs = pywt.wavedec(audio.audiodata, 'db4', level=5)
        # Apply thresholding to the wavelet coefficients
        threshold = np.std(coeffs[-1]) * 3  # Adjust the threshold as needed - threshold is set as twice the standard deviation of the last level coefficients. 
        coeffs = [pywt.threshold(c, threshold) for c in coeffs]
        denoised_audio = pywt.waverec(coeffs, 'db4')

        # melspec2 = librosa.feature.melspectrogram(
        #                 y=denoised_audio, sr=audio.samplerate, n_fft=self.kwargs.get('n_fft', 2048),hop_length=self.kwargs.get('hop_length',512),
        #                 window=self.kwargs.get('window','hann'),center=self.kwargs.get('center',True),pad_mode=self.kwargs.get('pad_mode','constant'),
        #                 power=self.kwargs.get('power',2.0)
        #             )
        # #fn2 = audio.sourcefile + "denoise"
        # fn2 = str(audio.label) + "_" + audio.sourcefile[:-4] + "noi"
        # savemelspec([melspec2,fn2,audio.samplerate])

        # wafn2 = "C:/Users/Alliewya/Documents/Cough Monitor/Spectrograms/Denoise_2/" + fn2 + ".wav"
        # sf.write(wafn2, audio.audiodata, audio.samplerate)

        feats = coeffs[0].ravel()
        return {'feature':feats, 'label':audio.label}


class WaveletScatterFeatures(FeaturesBase):

    def __init__(self, **kwargs):
            self.kwargs = kwargs
        

    def features_from_dataset2(self,dataset, **kwargs):
        sr = 22050
        framelength = sr // 2
        hop_lenght = sr // 4
        
        frames = librosa.util.frame(dataset.samples[0], frame_length=framelength, hop_length=hop_lenght)
        features = np.empty


    def features_from_dataset(self, dataset,**kwargs):

        features = None
        labels = None
        return features,labels

    def single_features(self, audiosample=None,audioframe = None, **kwargs):
        features = None
        return features

    def single_features_from_audio(self, audiofilepath, **kwargs):
        audiosample = AudioSample(path=audiofilepath, samplerate=self.kwargs.get('samplerate',22050))
        features = np.empty((0, kwargs.get('n_mfcc', 20)))
        np.append(features, librosa.feature.mfcc(
                y=audiosample.audiodata, sr=audiosample.samplerate, n_mfcc=n_mfcc
            ), axis=0)
        features = features.ravel()
        return features

    def process_sample(self, audio):

        kwargs = self.kwargs
        # mfcc = librosa.feature.mfcc(
        #     y=audio.audiodata, sr=audio.samplerate, n_mfcc=n_mfcc
        # )
        sample_rate = 22050
        # J = 6  # The maximum scale of the scattering transform (2**J should be smaller than the signal length)
        # Q = 1  # The number of wavelets per octave
        J = int(kwargs.get('wav_scat_j', 6)) 
        Q = int(kwargs.get('wav_scat_q',1))

        T = len(audio.audiodata)
        # print("Scatter 1")
        scattering = Scattering1D(J, T, Q)
        # print("Scatter 2")
        features = scattering(audio.audiodata)
        # print("Scatter 3")
        


        # features = features.ravel()
        
        # #AVG Fatures
        # #avg_features = np.mean(features, axis=1)
        # avg_features = np.mean(features)
        # avg_features = avg_features.ravel()
        # std_features = np.std(features)
        # std_features = std_features.ravel()

        # feats = np.concatenate((avg_features,std_features))


        # print("Scatter 4")

        feats = self.calculate_summary_features(features)
        #print(feats)
        #print(type(feats))
        feats = np.array(feats)
        feats = np.nan_to_num(feats)
        feats = feats.ravel()

        #print(contains_nan) 
        return {'feature':feats, 'label':audio.label}

    def calculate_summary_features(self, scattering_coeffs):
        features = []

        if scattering_coeffs.ndim == 1:
            scattering_coeffs = scattering_coeffs.reshape(1, -1)

        for i in range(scattering_coeffs.shape[0]):
            # # Calculate energy
            # energy = np.sum(np.abs(scattering_coeffs[i]) ** 2)

            # # Calculate entropy
            # entropy = 0.0
            # if np.any(scattering_coeffs[i] != 0):
            #     entropy = -np.sum(np.abs(scattering_coeffs[i]) ** 2 * np.log(np.abs(scattering_coeffs[i]) ** 2))
            # entropy = np.nan_to_num(entropy)

            # Calculate skewness
            skewness = skew(scattering_coeffs[i])

            # Calculate kurtosis
            kurtosis_val = kurtosis(scattering_coeffs[i])

            # Calculate variance
            variance = np.var(scattering_coeffs[i])

            # Calculate maximum
            maximum = np.max(scattering_coeffs[i])

            # Calculate minimum
            minimum = np.min(scattering_coeffs[i])

            # Calculate mean
            mean = np.mean(scattering_coeffs[i])

            # Calculate standard deviation
            std_dev = np.std(scattering_coeffs[i])

            #features.append([energy, entropy, skewness, kurtosis_val, variance, maximum, minimum, mean, std_dev])
            features.append([ skewness, kurtosis_val, variance, maximum, minimum, mean, std_dev])

        return features


class SpectralCentroidFeatures(FeaturesBase):
    """Spectral Centroid features represent the weighted mean of the frequencies in an audio signal."""

    def __init__(self, **kwargs):
            self.kwargs = kwargs

    def single_features(self, audio, **kwargs):
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio.audiodata, sr=audio.samplerate, **kwargs
        )
        return spectral_centroids

    def process_sample(self, audio):
        kwargs = self.kwargs
        n_fft = int(kwargs.get('n_fft', 2048))
        hop_length = int(kwargs.get('hop_length', 512))
        win_length = kwargs.get('win_length', None)
        if(win_length != None):
            win_length = int(win_length)
        window = kwargs.get('window', 'hann')
        center = bool(kwargs.get('center', True))
        pad_mode = kwargs.get('pad_mode', 'constant')

        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio.audiodata, sr=audio.samplerate, n_fft=n_fft, hop_length=hop_length,
            win_length=win_length, window=window, center=center, pad_mode=pad_mode
        )
        spectral_centroids = spectral_centroids.ravel()

        return {'feature': spectral_centroids, 'label': audio.label}


class SpectralBandwidthFeatures(FeaturesBase):
    """Spectral Bandwidth features measure the width of the frequency distribution in an audio signal."""
    def single_features(self, audio, **kwargs):
        spectral_bandwidths = librosa.feature.spectral_bandwidth(
            y=audio.audiodata, sr=audio.samplerate, **kwargs
        )
        return spectral_bandwidths

    def process_sample(self, audio):
        kwargs = self.kwargs
        n_fft = int(kwargs.get('n_fft', 2048))
        hop_length = int(kwargs.get('hop_length', 512))
        win_length = kwargs.get('win_length', None)
        if(win_length != None):
            win_length = int(win_length)
        window = kwargs.get('window', 'hann')
        center = bool(kwargs.get('center', True))
        pad_mode = kwargs.get('pad_mode', 'constant')
        centroid = kwargs.get('centroid', None)
        norm = kwargs.get('norm', False)

        spectral_bandwidths = librosa.feature.spectral_bandwidth(
            y=audio.audiodata, sr=audio.samplerate, n_fft=n_fft, hop_length=hop_length,
            win_length=win_length, window=window, center=center, pad_mode=pad_mode,
            centroid=centroid, norm=norm
        )
        spectral_bandwidths = spectral_bandwidths.ravel()

        return {'feature': spectral_bandwidths, 'label': audio.label}


class SpectralContrastFeatures(FeaturesBase):
    """Spectral Contrast features represent the difference in amplitude between peaks and valleys in an audio signal."""
    def single_features(self, audio, **kwargs):
        spectral_contrasts = librosa.feature.spectral_contrast(
            y=audio.audiodata, sr=audio.samplerate, **kwargs
        )
        return spectral_contrasts

    def process_sample(self, audio):
        kwargs = self.kwargs
        n_bands = int(kwargs.get('n_bands', 6))
        fmin = float(kwargs.get('fmin', 200.0))
        #fmax = float(kwargs.get('fmax', 6000.0))
        hop_length = int(kwargs.get('hop_length', 512))
        n_fft = int(kwargs.get('n_fft', 2048))
        win_length = kwargs.get('win_length', None)
        if(win_length != None):
            win_length = int(win_length)
        window = kwargs.get('window', 'hann')
        center = bool(kwargs.get('center', True))
        pad_mode = kwargs.get('pad_mode', 'constant')
        #freqs = kwargs.get('freqs', None)
        #harmonics = bool(kwargs.get('harmonics', False))
        #norm = bool(kwargs.get('norm', False))

        spectral_contrasts = librosa.feature.spectral_contrast(
            y=audio.audiodata, sr=audio.samplerate, n_bands=n_bands, fmin=fmin,
            hop_length=hop_length, n_fft=n_fft, win_length=win_length, window=window,
            center=center, pad_mode=pad_mode, 
        )
        spectral_contrasts = spectral_contrasts.ravel()

        return {'feature': spectral_contrasts, 'label': audio.label}


class SpectralRolloffFeatures(FeaturesBase):
    """Spectral Rolloff features represent the frequency below which a specified percentage of the total spectral energy lies."""
    def single_features(self, audio, **kwargs):
        spectral_rolloffs = librosa.feature.spectral_rolloff(
            y=audio.audiodata, sr=audio.samplerate, **kwargs
        )
        return spectral_rolloffs

    def process_sample(self, audio):
        kwargs = self.kwargs

        roll_percent = float(kwargs.get('roll_percent', 0.85))
        hop_length = int(kwargs.get('hop_length', 512))
        n_fft = int(kwargs.get('n_fft', 2048))
        win_length = kwargs.get('win_length', None)
        if(win_length != None):
            win_length = int(win_length)
        window = kwargs.get('window', 'hann')
        center = bool(kwargs.get('center', True))
        pad_mode = kwargs.get('pad_mode', 'reflect')

        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio.audiodata, sr=audio.samplerate, roll_percent=roll_percent, hop_length=hop_length,
            n_fft=n_fft, win_length=win_length, window=window, center=center, pad_mode=pad_mode
        )
        spectral_rolloff = spectral_rolloff.ravel()

        return {'feature': spectral_rolloff, 'label': audio.label}


class ChromaFeatures(FeaturesBase):
    """Chroma features represent the 12 different pitch classes in music."""
    def single_features(self, audio, **kwargs):
        chroma_features = librosa.feature.chroma_stft(
            y=audio.audiodata, sr=audio.samplerate, **kwargs
        )
        return chroma_features

    def process_sample(self, audio):
        kwargs = self.kwargs

        if(kwargs.get("tuning",440)):
            tune= None
        else:
            tune = kwargs.get("tuning",400)

        chroma_features = librosa.feature.chroma_stft(
            y=audio.audiodata, sr=audio.samplerate, n_chroma=int(kwargs.get('n_chroma', 12)),n_fft = int(kwargs.get('n_fft', 2048)),
            hop_length = int(kwargs.get('hop_length', 512)), tuning=tune
        )
        chroma_features = chroma_features.ravel()

        return {'feature': chroma_features, 'label': audio.label}


class ZeroCrossingRateFeatures(FeaturesBase):
    """Zero Crossing Rate features represent the rate at which a signal changes from positive to negative or vice versa."""
    def single_features(self, audio, **kwargs):
        zero_crossing_rates = librosa.feature.zero_crossing_rate(
            y=audio.audiodata, **kwargs
        )
        return zero_crossing_rates

    def process_sample(self, audio):
        kwargs = self.kwargs

        
        zero_crossing_rates = librosa.feature.zero_crossing_rate(
            y=audio.audiodata, frame_length=int(kwargs.get("frameLength", 2048)),hop_length=int(kwargs.get('hop_length', 512)),center=bool(kwargs.get('center', True))
        )
        zero_crossing_rates = zero_crossing_rates.ravel()

        return {'feature': zero_crossing_rates, 'label': audio.label}

class WaveletCepstralFeatures(FeaturesBase):

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        print("Wavelet Cepstral")
        print(kwargs)


    """ Wavelet Cepstral Coef"""
    def calculate_wcc(self, audio):
        kwargs = self.kwargs
        # Pre-emphasis
        pre_emphasis = 0.97
        emphasized_audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])

        # Frame the signal
        sr = 22050
        frame_length = int(sr * 0.025)  # 25ms
        hop_length = int(sr * 0.010)  # 10ms
        #frame_length = 1024
        #hop_length = 256

        frames = librosa.util.frame(emphasized_audio, frame_length=frame_length, hop_length=hop_length).T

        # Apply window function
        window = np.hamming(frame_length)
        windowed_frames = frames * window

        # Compute wavelet transform
        #wavelet = 'db4'
        #wavelet = 'haar'
        wavelet = kwargs.get("wavelet", 'haar')
        level = int(kwargs.get("level", 5))

        # Initialize WCC coefficients
        wcc = []

        # Compute Wavelet Cepstral Coefficients for each frame
        for frame in windowed_frames:
            # Apply wavelet transform
            coeffs = pywt.wavedec(frame, wavelet=wavelet, level=level)

            # Calculate subband energies
            subband_energies = [np.sum(np.square(subband)) for subband in coeffs]

            # Replace problematic values with zeros
            subband_energies = np.where(subband_energies == 0, 0, subband_energies)

            # Apply logarithm to subband energies
            log_energies = np.log(subband_energies)
            
            log_energies = np.nan_to_num(log_energies, nan=0.0)

            # Apply DCT
            n_ceps = 12  # Number of cepstral coefficients
            dct_coeffs = dct(log_energies, norm='ortho')[:n_ceps]

            # Append DCT coefficients to WCC
            wcc.append(dct_coeffs)

        wcc = np.array(wcc)

        # Apply DCT using scipy.fft.dct
        # n_ceps = 12
        # dct_filters = dct(np.eye(wcc.shape[1]), norm='ortho')
        # wcc = np.dot(wcc, dct_filters.T)


        # Keep the desired number of coefficients
        #wcc = wcc[:, :n_ceps]

        # Optional steps:
        # Compute delta and delta-delta coefficients
        #delta = librosa.feature.delta(wcc)
        #delta_delta = librosa.feature.delta(wcc, order=2)

        # Perform normalization
        wcc_mean = np.mean(wcc, axis=0)
        wcc_std = np.std(wcc, axis=0)

        # Check for zero standard deviation to avoid divide by zero
        wcc_std[wcc_std == 0.0] = 1e-8

        wcc_normalized = (wcc - wcc_mean) / wcc_std

        wcc_normalized = np.nan_to_num(wcc_normalized,nan=0)

        return wcc_normalized


    def process_sample(self, audio):
        
        kwargs = self.kwargs
        
        wavelet_cc = self.calculate_wcc(audio.audiodata)
        wavelet_cc = wavelet_cc.ravel()

        return {'feature': wavelet_cc, 'label': audio.label}



#Multi-Features Extractor
class MultiExtractor(FeaturesBase):
    def __init__(self, extractors):
        self.extractors = extractors

    def features_from_dataset(self, dataset, **kwargs):
        features = []
        labels = []
        for audio in dataset.samples:
            sample_features = self.single_features(audio, **kwargs)
            features.append(sample_features['feature'])
            labels.append(sample_features['label'])
        return features, labels

        
    def single_features(self, audiosample=None,audioframe = None, **kwargs):
        audio_features = []
        for extractor in self.extractors:
            if(audiosample):
                sample_result = extractor.process_sample(audiosample)
            else:
                a = AudioSample()
                a.audiodata = audioframe
                a.label = "2"
                a.samplerate = 22050
                sample_result = extractor.process_sample(a)
            features = sample_result['feature']
            audio_features.append(features)
            
        combined_features = np.concatenate(audio_features)

        contains_nan = np.isnan(combined_features).any()
        if contains_nan:
            print("NaN values detected in features")
        return combined_features

    def single_features_from_audio(self, audiofilepath, **kwargs):
        features = {}
        for extractor in self.extractors:
            extractor_features = extractor.single_features_from_audio(audiofilepath, **kwargs)
            features.update(extractor_features)
        return features

    def process_sample(self, audio):
        audio_features = []
        for extractor in self.extractors:
            #print(type(extractor))
            sample_result = extractor.process_sample(audio)
            features = sample_result['feature']
            audio_features.append(features)

        combined_features = np.concatenate(audio_features)

        contains_nan = np.isnan(combined_features).any()
        if contains_nan:
            print("NaN values detected in features")

        return {'feature': combined_features, 'label': audio.label}



class FeaturesFactory():
    """Features class that returns a feature extractor"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.extractors = []
        if bool(self.kwargs["mfcc"]['enable']):
            self.extractors.append(MFCCFeatures(**kwargs['mfcc']))
        elif bool(self.kwargs["mel_spec"]['enable']):
           self.extractors.append(MelSpectrogramFeatures(**kwargs['mel_spec']))

    def extract_features(self, dataset=None, audiosamp=None,**kwargs):
        if(dataset):
            bing = 1
            print(type(dataset.samples))
            with multiprocessing.Pool(processes=8) as pool:
                results = pool.starmap(
                    self.process_sample_multi_extractor,
                    [(audio, bing) for audio in dataset.samples]
                )
            features = [x['feature'] for x in results]
            labels = [x['label'] for x in results]
            # if(self.kwargs["concat"]=="enable"):
            #     features = [np.concatenate(sublist) for sublist in features]

            ############ Make Concac option
            features = [np.concatenate(sublist) for sublist in features]

            features = np.asarray(features)
            print("Features Shape")
            print(features.shape)
            features = features.reshape((features.shape[0], -1))
            print("Features Reshape")
            print(features.shape)
        else:
            bing = 1
            print(type(dataset.samples))
            with multiprocessing.Pool() as pool:
                results = pool.starmap(
                    self.process_sample_multi_extractor,
                    (audiosamp,bing)
                )
            features = [x['feature'] for x in results]
            labels = [x['label'] for x in results]
            # if(self.kwargs["concat"]=="enable"):
            #     features = [np.concatenate(sublist) for sublist in features]

            ############ Make Concac option
            features = [np.concatenate(sublist) for sublist in features]

            features = np.asarray(features)
            print("Features Shape")
            print(features.shape)
            features = features.reshape((features.shape[0], -1))
            print("Features Reshape")
            print(features.shape)
            return 
        return features, labels

    def process_sample_multi_extractor(self, audio, kwargs):
        bing = kwargs
        features = []
        label = audio.label
        for i, extractor in enumerate(self.extractors):
            f = extractor.process_sample(audio,bing)
            if(label != f['label'] ):
                print(label)
                print(f['label'])
                print("Not matching labels")
                raise ValueError("Label mismatch between different feature extractors")
            features.append(f['feature'])
            label = f['label']
        return {'feature':features,'label':label}

# ========================================================
#               Feature Preprocessing
# ========================================================


class FeaturePreprocessorBase(ABC):
    @abstractmethod
    def preprocess_features(self, features):
        pass

    @abstractmethod
    def transform_features(self,features):
        pass


class MinMaxScalerPreprocessor(FeaturePreprocessorBase):
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def preprocess_features(self, features):
        print("Min Max Scaler")
        self.scaler = self.scaler.fit(features)
        scaled_features = self.scaler.transform(features)
        return scaled_features
    
    def transform_features(self,features):
        scaled_features = self.scaler.transform(features) 
        return scaled_features

class PCAFeaturePreprocessor(FeaturePreprocessorBase):
    def __init__(self, n_components=128):
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)

    def preprocess_features(self, features):
        print("PCA")
        #print(features[0].shape)
        #flattened_features = np.concatenate(features, axis=0)
        self.pca = self.pca.fit(features)
        transformed_features = self.pca.transform(features)
        print(transformed_features[0].shape)
        return transformed_features

    def transform_features(self,features):
        transformed_features = self.pca.transform(features)
        print(transformed_features[0].shape)
        return transformed_features




# =======================================================
#               Model Section
# =======================================================

class ModelBase():
    """An abstract base class for all pipeline models"""

    # @abstractmethod
    # def fit(self, X, y):
    #     """"""
    #     pass

    # @abstractmethod
    # def predict(self, X, y):
    #     """"""
    #     pass

    # @abstractmethod
    # def evaluate(self, X, y):
    #     """"""
    #     pass


class KNNModel(ModelBase, KNeighborsClassifier):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        super().__init__(
            n_neighbors=int(kwargs["k"]),
            algorithm=kwargs["algorithm"],
            leaf_size=int(kwargs["leaf_size"]),
            metric=kwargs["metric"],
            metric_params={"p": int(kwargs["metric_power"])},
            weights=kwargs["weights"],
        )



class SVMModel(ModelBase, SVC):
    def __init__(self, **kwargs):
        super().__init__(
            C=float(kwargs["c"]),
            kernel=kwargs["kernel"],
            degree=int(kwargs["degree"]),
            gamma=kwargs["gamma"],
            shrinking=kwargs["shrinking"],
            probability=kwargs["probability"],
            tol=float(kwargs["tol"]),
            max_iter=kwargs["max_iter"]
        )
        if kwargs["max_iter"] == "-1":
            self.max_iter = -1
        else:
            self.max_iter = int(kwargs["max_iter"])

class AdaBoostModel(ModelBase, AdaBoostClassifier):
    def __init__(self, **kwargs):
        base_estimator = self.create_base_estimator(kwargs)
        if kwargs['ada_random_state'] == "None":
            ada_random_state = None
        else:
            ada_random_state = int(kwargs['ada_random_state'])
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=int(kwargs["ada_n_estimators"]),
            learning_rate=float(kwargs["ada_learning_rate"]),
            algorithm=kwargs["ada_algorithm"],
            random_state=ada_random_state,
        )

    @staticmethod
    def create_base_estimator(kwargs):
        if kwargs["estimator"] == "None":
            return None
        elif kwargs["estimator"] == "DecisionTreeClassifier":
            #dt_params = kwargs["ada_dt_config"]
            return DecisionTreeClassifier(
                max_depth=None if kwargs["ada_dt_max_depth"] == "None" else int(kwargs["ada_dt_max_depth"] ),
                min_samples_split=int(kwargs["ada_dt_min_samples_split"]),
                min_samples_leaf=int(kwargs["ada_dt_min_samples_leaf"]),
                criterion=kwargs["ada_dt_criterion"],
                max_leaf_nodes=None if kwargs["ada_decisiontree_max_leaf_nodes"] == "None" else int(kwargs["ada_decisiontree_max_leaf_nodes"]),
                splitter=kwargs["ada_decision_tree_splitter"]
            )
        elif kwargs["estimator"] == "SVC":
            #svm_params = kwargs["ada_svm_config"]
            return SVC(
                C=float(kwargs["ada_svm_c"]),
                kernel=kwargs["ada_svm_kernel"],
                degree=int(kwargs["ada_svm_degree"]),
                gamma=kwargs["ada_svm_gamma"],
                shrinking=kwargs["ada_svm_shrinking"],
                probability=True,
                tol=float(kwargs["ada_svm_tol"]),
                max_iter=int(kwargs["ada_svm_max_iter"]),
            )


class LogisticRegressionModel(ModelBase, LogisticRegression):
    def __init__(self, **kwargs):
        super().__init__(
            penalty=kwargs["penalty"],
            C=float(kwargs["C"]),
            solver=kwargs["solver"],
            fit_intercept=kwargs["fit_intercept"],
            max_iter=int(kwargs["max_iter"]),
            tol=float(kwargs["tol"]),
        )


class DecisionTreeModel(ModelBase, DecisionTreeClassifier):
    def __init__(self, **kwargs):
        super().__init__(
            max_depth=None if kwargs["max_depth"] == "None" else int(kwargs["max_depth"]),
            min_samples_split=int(kwargs["min_samples_split"]),
            min_samples_leaf=int(kwargs["min_samples_leaf"]),
            criterion=kwargs["criterion"],
            max_leaf_nodes=None if kwargs["max_leaf_nodes"] == "None" else int(kwargs["max_leaf_nodes"]),
            splitter=kwargs["splitter"],
        )


class GMMHMMModel(ModelBase):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.models = []

    def fit(self, X, y):
        for label in np.unique(y):
            X_label = [X[i] for i in range(len(X)) if y[i] == label]
            #model = self.train_hmm_gmm_model(X_label, int(self.kwargs['n_components']), self.kwargs['n_mix'])
            model = self.train_hmm_gmm_model(X_label)
            self.models.append(model)
        return self

    def train_hmm_gmm_model(self, X, n_states=2, n_mixtures=2, n_iter=100):
        model = hmm.GMMHMM(n_components=n_states)
        model.fit(X)
        return model

    def predict(self, X):
        max_logprob = float('-inf')
        best_label = -1
        for i, model in enumerate(self.models):
            logprob = model.score(X)
            if logprob > max_logprob:
                max_logprob = logprob
                best_label = i
        print(best_label)
        return best_label

    def predict_proba(self, X):
        max_logprob = float('-inf')
        best_label = -1
        for i, model in enumerate(self.models):
            logprob = model.score(X)
            if logprob > max_logprob:
                max_logprob = logprob
                best_label = i
        return best_label

    def score(self, X, y):
        print("Scrore start")
        #y_pred = [self.predict(x) for x in X]
        y_pred = [self.predict(x.reshape(1,-1)) for x in X]
        accuracy = accuracy_score(y, y_pred)
        return accuracy

class ClassifierFactory:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def create_classifier(self):
        if bool(self.kwargs["knn"]['enable']):
            knn_params = self.kwargs["knn"]
            return KNNModel(**knn_params)
        elif bool(self.kwargs["svm"]['enable']):
            svm_params = self.kwargs["svm"]
            return SVMModel(**svm_params)
        elif bool(self.kwargs["adaboost"]['enable']):
            adaboost_params = self.kwargs["adaboost"]
            return AdaBoostModel(**adaboost_params)
        elif  bool(self.kwargs["logistic_regression"]['enable']):
            logreg_params = self.kwargs["logistic_regression"]
            return LogisticRegressionModel(**logreg_params)
        elif bool(self.kwargs["decision_tree"]['enable']):
            dt_params = self.kwargs["decision_tree"]
            return DecisionTreeModel(**dt_params)
        elif bool(self.kwargs["GMMHMM"]['enable']):
            GMMHMM_params = self.kwargs['GMMHMM']
            return GMMHMMModel(**GMMHMM_params)
        else:
            return None


                    

#a = MFCCFeatures().features_from_dataset()


#Scoring
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score
import time

def common_scores(clf, X_test, y_test):
    print("Scores Start")
    scores = {}
    # Calculate the accuracy score
    #print("1")
    #print(X_test.shape)
    scores["accuracy"] = clf.score(X_test, y_test)
    start_time = time.time()                        
    y_pred = clf.predict(X_test)
    #print(y_test)
    #print(y_pred)
    end_time = time.time()
    scores["timetaken"] = end_time - start_time
    if isinstance(clf, GMMHMMModel):
        return scores
    print("2")
    scores["f1"] = f1_score(y_test, y_pred, average='binary', pos_label='0')
    scores["precision"] = precision_score(y_test, y_pred, average='binary', pos_label='0')
    scores["recall"] = recall_score(y_test, y_pred, average='binary', pos_label='0')
    print("3")
    #y_score = clf.predict_proba(X_test)
    y_score = clf.predict_proba(X_test)[:,1]
    scores["roc_auc"] = roc_auc_score(y_test, y_score)
    #print("4")
    #scores["confusion_matrix"] = confusion_matrix(y_test, y_pred, labels=['1', '0']).tolist()
    scores["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
    #scores["confusion_matrix"]= scores["confusion_matrix"]
    print(scores)
    return scores


from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold, StratifiedKFold

class CrossVal():
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def do_cross_validate(self,clfin, clfparm, X, y):
        print("Cross Val 1")
        params = self.kwargs
        #print(self.kwargs)
        if(bool(params['kfold']['enable'])):
            print("Cross Val 2")
            spram = params['kfold']
            if spram['kfold_random_state'] == 'None':
                random_state = None
            else:
                random_state = int(spram['kfold_random_state'])
            cv = KFold(n_splits=int(spram['kfold_n_splits']),shuffle=bool(spram['kfold_shuffle']),random_state=random_state)
            print("Cross Val 3")
        elif(bool(params['stratifiedkfold']['enable'])):
            spram = params['stratifiedkfold']
            if spram['stratifiedkfold_random_state'] == 'None':
                random_state = None
            else:
                random_state = int(spram['stratifiedkfold_random_state'])
            cv = StratifiedKFold(n_splits=int(spram['stratifiedkfold_n_splits']),shuffle=bool(spram['stratifiedkfold_shuffle']),random_state=random_state)
        print("Cross Val 4")
        print(clfin)
        #modelfactory = ClassifierFactory(**clfin)
        #mdl = modelfactory.create_classifier()

        if bool(clfparm["knn"]['enable']):  
            knn_params = {
            "n_neighbors": int(clfparm["knn"]["knn_k"]),
            "algorithm": clfparm["knn"]["knn_algorithm"],
            "leaf_size": int(clfparm["knn"]["knn_leaf_size"]),
            "metric": clfparm["knn"]["knn_metric"],
            "metric_params": {"p": int(clfparm["knn"]["knn_metric_power"])},
            "weights": clfparm["knn"]["knn_weights"]
            }
            mdl = KNeighborsClassifier(**knn_params)
        # elif bool(clfparm["svm"]['enable']):
        #     svm_params = self.kwargs["svm"]
        #     return SVMModel(**svm_params)
        # elif bool(clfparm["adaboost"]['enable']):
        #     adaboost_params = self.kwargs["adaboost"]
        #     return AdaBoostModel(**adaboost_params)
        # elif  bool(clfparm["logistic_regression"]['enable']):
        #     logreg_params = self.kwargs["logistic_regression"]
        #     return LogisticRegressionModel(**logreg_params)
        # elif bool(clfparm["decision_tree"]['enable']):
        #     dt_params = self.kwargs["decision_tree"]
        #     return DecisionTreeModel(**dt_params)
        # elif bool(clfparm["GMMHMM"]['enable']):
        #     GMMHMM_params = self.kwargs['GMMHMM']
        # else:
        #     return None

        scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        cv_results = cross_validate(estimator=mdl, X=X, y=y, cv=cv, n_jobs=-1, verbose=1, scoring=scoring)
        #clfin.set_params(**clfparm)
        #cv_results = cross_validate(estimator=clfin, X=X, y=y, cv=cv, n_jobs=-1, verbose=1)
        # h = KNeighborsClassifier()
        # print(type(h))
        # print(dir(h))
        # print(h.get_params(deep=True))
        # print(type(clfin))
        # print(dir(clfin))
        # #print(clfin.get_params(deep=True))
        #cv_results = cross_validate(estimator=KNeighborsClassifier(), X=X, y=y, cv=cv, n_jobs=-1, verbose=1)
        print("Cross Val 5")
        print(cv_results)
        return cv_results



class Predictor():

    def __init__(self,featureeex, clf, **kwargs):
         self.featureextractor = featureeex
         self.classifier = clf

    def make_prediction(self,audio):
        features = self.featureextractor.single_features(audioframe=audio)
        features = features.reshape(1, -1)
        prediction = self.classifier.predict(features)
        return prediction
        
    def make_prediction_probability(self, audio):
        features = self.featureextractor.single_features(audioframe=audio)
        features = features.reshape(1, -1)
        prediction = self.classifier.predict(features)
        try:
            probability = self.classifier.predict_proba(features)
            return prediction, probability
        except AttributeError:
            raise Exception("Classifier doesn't support probability")


def tune_hyperparameters( clf, param_grid, X, y, cv=5):
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=cv,n_jobs=10)
    grid_search.fit(X, y)
    return grid_search


###################
# Config Handler 
###################




class ConfigProcessor:
    def process_config(self, data):
        feature_extractor = FeatureExtractor()
        dataset_processor = DatasetProcessor()
        audio_preprocessor = PreprocessingProcessor()
        feature_preprocessor = FeaturePreprocessor()  # Create an instance of the feature preprocessing class
        classifier_factory = ClassifierFactory(**data['classifier'])
        # evaluation_processor = EvaluationProcessor()
        predictor_saver = PredictorSaver()

        #Feature Extractor Creation
        featureconfig = data['features']
        print(featureconfig)
        #a = feature_extractor.create_mel_spectrogram_feature(**featureconfig['mel_spec'])
        #b = feature_extractor.create_mfcc_feature(**featureconfig['mfcc'])
        extractor = feature_extractor.create_extractor_from_config(featureconfig)

        #Dataset Loading
        dataset = dataset_processor.load_dataset(data['datasetpickle'])
        print(data['dataset'])
        dataset_processor.process_dataset(dataset, data['dataset'])
        try:
            print(dataset.descriptor)
        except:
            print("Older Dataset without descriptor")
        print("Dataset processed")
    
        #Audio Preprocessing
        dataset = audio_preprocessor.preprocessing_from_config(dataset,data['preprocessing'])
        
        print("Preprocessing processed")

        #Feature Extracting
        # mfc, labels = b.features_from_dataset_multi(dataset)
        # print("Mfcc extracted")
        # scatter, labels2 = feature_extractor.create_wavelet_scatter_features(**data['features']).features_from_dataset_multi(dataset)
        # print("Scatter extracted")
        features, labels = extractor.features_from_dataset_multi(dataset)

        #for i in range(len(features)):
        #    features[i] = np.array([np.random.random()])



        print("Features Extracted " + str(features[0].shape))


        # Feature Preprocessing
        #preprocessed_features = feature_preprocessor.featurepreprocessing_from_config(features,data['featurepreprocessing'])  # Call feature preprocessing method
        

        #Classifier Create
        mdl = classifier_factory.create_classifier()
        print("Mdl Created")
        
        #Evaluation
        scores = self.evaluate_model(mdl,data, data['evaluation'], features, labels)
        #scores = self.evaluate_model(mdl,data, data['evaluation'], preprocessed_features, labels)

        #Hyperparamter search enabled
        if bool(data['classifier']['knn']['tune_hyperparameters']) or bool(data['classifier']['svm']['tune_hyperparameters']):
            print("hyperparameters")
            self.tune_hyperparameters_config( mdl, data, features, labels)

        predictor = predictor_saver.save_predictor(mdl, extractor)
        return {"Status": "Success", "Scores": scores, "Config": data, "Timestamp": datetime.now().isoformat()}


    def evaluate_model(self, mdl, data,evaluation_data,features, labels):
        scores = {}
        if evaluation_data['train_test_split']['enable']:
            scores = self.evaluate_train_test(mdl, data, evaluation_data['train_test_split'],features, labels)
        elif evaluation_data['kfold']['enable']:
            scores = self.evaluate_kfold(mdl,data, evaluation_data, features, labels)
        elif evaluation_data['stratifiedkfold']['enable']:
            scores = self.evaluate_stratified_kfold(mdl, evaluation_data['stratifiedkfold'],features, labels)
        return scores

    def evaluate_train_test(self, mdl, data, train_test_data,features, labels):
        feature_preprocessor = FeaturePreprocessor()
        
        #feature_preprocessor.featurepreprocessing_transform
        params = train_test_data
        print(train_test_data)
        if params['random_state'] == "None":
            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=float(params['test_size']),
                                                                shuffle=params['shuffle'])
            X_train = feature_preprocessor.featurepreprocessing_from_config(X_train,data['featurepreprocessing'])
            X_test = feature_preprocessor.featurepreprocessing_transform(X_test,data['featurepreprocessing'])
        else:
            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=float(params['test_size']),
                                                                shuffle=params['shuffle'],
                                                                random_state=int(params['random_state']))
            X_train = feature_preprocessor.featurepreprocessing_from_config(X_train,data['featurepreprocessing'])
            X_test = feature_preprocessor.featurepreprocessing_transform(X_test,data['featurepreprocessing'])
        # Fit the model and calculate scores
        clf = mdl.fit(X_train, y_train)
        print(clf.score(X_test,y_test))
        scores = common_scores(clf, X_test=X_test, y_test=y_test)

        return scores

    def evaluate_kfold(self, mdl, data,evaluationdata, features, labels):
        cv = CrossVal(**evaluationdata)
        print(data)
        print(evaluationdata)
        results = cv.do_cross_validate(clfin=mdl, clfparm=data['classifier'], X=features, y=labels)
        scores = {}

        scores['accuracy_mean'] = np.mean(results['test_accuracy'])
        scores['accuracy_standard_deviation'] = np.std(results['test_accuracy'])
        scores['precision_macro_mean'] = np.mean(results['test_precision_macro'])
        scores['precision_macro_standard_deviation'] = np.std(results['test_precision_macro'])
        scores['recall_macro_mean'] = np.mean(results['test_recall_macro'])
        scores['recall_macro_standard_deviation'] = np.std(results['test_recall_macro'])
        scores['f1_macro_mean'] = np.mean(results['test_f1_macro'])
        scores['f1_macro_standard_deviation'] = np.std(results['test_f1_macro'])
        scores['fit_time_mean'] = np.mean(results['fit_time'])
        scores['fit_time_standard_deviation'] = np.std(results['fit_time'])
        scores['score_time_mean'] = np.mean(results['score_time'])
        scores['score_time_standard_deviation'] = np.std(results['score_time'])


        scores['test_accuracy'] = results['test_accuracy'].tolist()
        scores['test_precision_macro'] = results['test_precision_macro'].tolist()
        scores['test_recall_macro'] = results['test_recall_macro'].tolist()
        scores['test_f1_macro'] = results['test_f1_macro'].tolist()
        scores['fit_time'] = results['fit_time'].tolist()
        scores['score_time'] = results['score_time'].tolist()
        return scores

    def evaluate_stratified_kfold(self, mdl, stratified_kfold_data, mfc, labels, scatter, labels2):
        cv = CrossVal(**stratified_kfold_data)
        results = cv.cross_validate(clf=mdl, X=mfc, y=labels)
        scores = {}
        scores['fit_time'] = results['fit_time'].tolist()
        scores['score_time'] = results['score_time'].tolist()
        return scores

    def tune_hyperparameters_config(self, clf, data, features, labels):
        if bool(data['classifier']['knn']['tune_hyperparameters']):
            params = data['evaluation']
            knn_param_grid = {
                "n_neighbors": list(range(1, 15)),
                "weights": ["uniform", "distance"],
                "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                "leaf_size": [10,20,30],
                
            }
            knn_param_grid = {
                "n_neighbors": list(range(1, 31)),
                "weights": ["uniform", "distance"],
                "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                "leaf_size": [20],
                "metric": ["euclidean", "manhattan", "chebyshev", "minkowski"],
                "metric_params": [None, {"p": 1}, {"p": 2}, {"p": 3}],
            }
            #"metric": ["euclidean", "manhattan", "chebyshev", "minkowski"],
            #    "metric_params": [None, {"p": 1}, {"p": 2}, {"p": 3}],
            X_train, X_test, y_train, y_test = train_test_split(features, labels, 
                    test_size=float(params['train_test']['test_size']), 
                    shuffle=params['train_test']['tts_shuffle'])
            mdl = tune_hyperparameters(KNeighborsClassifier(), knn_param_grid, X_train, y_train)
            scores = mdl.cv_results_
            #saving = ClassifierResults(data = str(scores))
            df = pd.DataFrame(scores)
            # Export the dataframe to a CSV file
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            fname = timestamp + "-result.csv"
            df.to_csv(fname, index=False)
            #saving.save()
        elif bool(data['classifier']['svm']['tune_hyperparameters']):
                params = data['evaluation']
                # svm_param_grid = {
                #     'C': [0.1, 1, 10, 100, 1000],
                #     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                #     'degree': [2, 3, 4, 5],  # Only used for the 'poly' kernel
                #     'gamma': ['scale', 'auto'] + list(np.logspace(-3, 2, 6)),  # 'scale', 'auto', and some values between 0.001 and 100
                #     'shrinking': [True, False],
                #     'probability': [True],
                #     'tol': [1e-4, 1e-3, 1e-2, 1e-1, 1],
                #     'max_iter': [-1, 100, 500, 1000, 5000],
                # }
                # svm_param_grid = {
                #     'C': [0.1, 1, 100, 1000],
                #     'kernel': ['linear', 'rbf',],
                #     'gamma': ['scale', 'auto'] + list(np.logspace(-3, 2, 6)),  # 'scale', 'auto', and some values between 0.001 and 100
                #     'shrinking': [True, False],
                #     'probability': [True],
                #     'tol': [1e-4, 1e-3, 1e-2, 1e-1, 1],
                #     'max_iter': [-1, 100, 500],
                # }
                svm_param_grid = {
                    'C': [0.1, 1, 100, 1000],
                    'kernel': ['linear', 'rbf',],
                    'gamma': ['scale', 'auto'] ,  # 'scale', 'auto', and some values between 0.001 and 100
                    'shrinking': [True, False],
                    'probability': [True],
                    'max_iter': [-1, 100, 500],
                }
                knn_param_grid = {
                    "n_neighbors": list(range(1, 15)),
                    "weights": ["uniform", "distance"],
                    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                    "leaf_size": [10, 20, 30],

                }
                # X_train, X_test, y_train, y_test = train_test_split(mfc3, labels2,
                #                                                     test_size=float(
                #                                                         params['train_test']['tts_test_size']),
                #                                                     shuffle=params['train_test']['tts_shuffle'])
                X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                                    test_size=float(
                                                                        params['train_test']['tts_test_size']),
                                                                    shuffle=params['train_test']['tts_shuffle'])
                mdl = tune_hyperparameters(
                    SVC(), svm_param_grid, X_train, y_train)
                scores = mdl.cv_results_
               # saving = ClassifierResults(
                #    data=str(scores))
                df = pd.DataFrame(scores)
                # Export the dataframe to a CSV file
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                fname = timestamp + "-result.csv"
                df.to_csv(fname, index=False)
                #saving.save()
        #mdl = tune_hyperparameters(clf, param_grid, X_train, y_train)
        return mdl


class FeatureExtractor:
    def create_extractor_from_config(self, featureconfig, **kwargs):
        extractor_classes = {
            'mfcc': MFCCFeatures,
            'mel_spec': MelSpectrogramFeatures,
            'wavedec': WaveletDecFeatures,
            'wav_scat': WaveletScatterFeatures,
            'spectral_centroid': SpectralCentroidFeatures,
            'spectral_bandwidth': SpectralBandwidthFeatures,
            'spectral_contrast': SpectralContrastFeatures,
            'spectral_rolloff': SpectralRolloffFeatures,
            'chroma_stft': ChromaFeatures,
            'zcr': ZeroCrossingRateFeatures,
            'wavelet_cepstral': WaveletCepstralFeatures,
        }

        enabled_extractors = []
        for feature_name, feature_params in featureconfig.items():
            enable = feature_params.get('enable', False)
            #feature_params.pop('enable')
            if enable and feature_name in extractor_classes:
                #print(feature_name)
                extractor_class = extractor_classes[feature_name]
                extractor = extractor_class(**feature_params)
                extractor.testfeaturekwargs(**kwargs)
                enabled_extractors.append(extractor)

        if len(enabled_extractors) == 0:
            raise ValueError("No feature extractor enabled.")
        elif len(enabled_extractors) > 1:
            return MultiExtractor(enabled_extractors)
        else:
            return enabled_extractors[0]

    # def create_feature_extractor(self, feature_name, **kwargs):
    #     extractor_classes = {
    #         'mfcc': MFCCFeatures,
    #         'mel_spec': MelSpectrogramFeatures,
    #         'wavedec': WaveletDecFeatures,
    #         'wav_scat': WaveletScatterFeatures,
    #         'spectral_centroid': SpectralCentroidFeatures,
    #         'spectral_bandwidth': SpectralBandwidthFeatures,
    #         'spectral_contrast': SpectralContrastFeatures,
    #         'spectral_rolloff': SpectralRolloffFeatures,
    #         'chroma': ChromaFeatures,
    #         'zero_crossing_rate': ZeroCrossingRateFeatures
    #     }

    #     if feature_name in extractor_classes:
    #         extractor_class = extractor_classes[feature_name]
    #         extractor = extractor_class(**kwargs)
    #         return extractor
    #     else:
    #         return None


class DatasetProcessor:
    def load_dataset(self, dataset_pickle):
        path = os.path.join("app", "static", "datasetpickles", dataset_pickle)
        with open(path, 'rb') as f:
            dataset = pickle.load(f)
        return dataset

    def process_dataset(self, dataset, dataset_config):
        dataset.add_labels_to_audiosamps()
        if bool(dataset_config['processing']['enable_normalize']):
            print("Norm")
            dataset.normalize()
        if bool(dataset_config['processing']['enable_segment']):
            dataset.segment()
        if bool(dataset_config['processing']['enable_trim']):
            dataset.trim_audio()
        if bool(dataset_config['fixed']['enable']):
            dataset.fix_length(length=dataset_config['fixed']['length'])
            print(dataset.samples[1].audiodata.shape)
        elif bool(dataset_config['frame']['enable']):
            dataset.create_frames(frame_size=int(dataset_config['frame']['length']),
                                  hop_length=int(dataset_config['frame']['hop_length']))


class PreprocessingProcessor:

    def preprocessing_from_config(self, dataset, preprocessing_config):
        if bool(preprocessing_config['wavelet_denoise']['enable']):
            print(preprocessing_config)
            dataset = WaveletDenoise().preprocess_audio_dataset_multi(dataset,**preprocessing_config['wavelet_denoise'])
            print("Wavelet Denoise Compete")
        return dataset
    

class FeaturePreprocessor:

    def featurepreprocessing_from_config(self, features, featurepreprocessing_config):
        #featurepreprocessing_config['PCA']['enable'] = True
        print(featurepreprocessing_config)
        if bool(featurepreprocessing_config['min_max']['enable']):
            self.minmax = MinMaxScalerPreprocessor()
            features = self.minmax.preprocess_features(features)
        if bool(featurepreprocessing_config['pca']['enable']):
            print(featurepreprocessing_config)
            ### ADD HERE RETURNING OF FIT PCA FOR REUSE IN PREDICTOR
            self.pca = PCAFeaturePreprocessor(n_components=int(featurepreprocessing_config['pca']['n_components']))
            features = self.pca.preprocess_features(features)
        #features = PCAFeaturePreprocessor().preprocess_features(features)
        return features

    def featurepreprocessing_transform(self,features,featurepreprocessing_config):
        print("transform_features")
        if bool(featurepreprocessing_config['min_max']['enable']):
            features = self.minmax.transform_features(features)
        if bool(featurepreprocessing_config['pca']['enable']):
            print(featurepreprocessing_config)
            ### ADD HERE RETURNING OF FIT PCA FOR REUSE IN PREDICTOR
            features = self.pca.transform_features(features)
        #features = PCAFeaturePreprocessor().preprocess_features(features)
        return features
    

# class EvaluationProcessor:
#     def evaluate_model(self, mdl, evaluation_data, mfc, labels):
#         scores = {}
#         if evaluation_data['train_test']['enable']:
#             scores = self.evaluate_train_test(mdl, evaluation_data['train_test'], mfc, labels)
#         elif evaluation_data['kfold']['enable']:
#             scores = self.evaluate_kfold(mdl, evaluation_data['kfold'], mfc, labels)
#         elif evaluation_data['stratifiedkfold']['enable']:
#             scores = self.evaluate_stratified_kfold(mdl, evaluation_data['stratifiedkfold'], mfc, labels)
#         return scores

#     def evaluate_train_test(self, mdl, train_test_data, mfc, labels):
#         pass
#         # Code for evaluating the model using train-test split

#     def evaluate_kfold(self, mdl, kfold_data, mfc, labels):
#         pass
#         # Code for evaluating the model using k-fold cross-validation

#     def evaluate_stratified_kfold(self, mdl, stratified_kfold_data, mfc, labels):
#         pass
#         # Code for evaluating the model using stratified k-fold cross-validation


class PredictorSaver:
    def save_predictor(self, mdl, feature_extractor):
        predictor = Predictor(featureeex=feature_extractor, clf=mdl)
        with open("predictor4.pickle", "wb") as file:
            pickle.dump(predictor, file)
        return predictor
    


#########################
# Helper / Misc Functions
############################


def savemelspec(params):
    spec, name, sr = params
    # Convert amplitude to decibels
    mel_spectrogram_db = librosa.power_to_db(spec, ref=np.max)

    # Plot the Mel spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()

    # Specify the folder path to save the images
    folder_path = 'C:/Users/Alliewya/Documents/Cough Monitor/Spectrograms/Denoise_2'
    os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist

    # Save the Mel spectrogram in the folder with the specified filename
    fn = os.path.join(folder_path, name + "" + ".png")
    plt.savefig(fn)
    plt.close()


        # #Saving MelSpec ###############################################################
        # # Save features and labels to a pickle file
        # with open('features_labels.pkl', 'wb') as f:
        #     pickle.dump((features, labels), f)

        # # # for index, feats in enumerate(features):
        # # #     fn = str(labels[index])  + "_" + str(index)
        # # #     savemelspec(feats,fn)

        # params_list = [(feat, str(labels[i]) + "_" + str(i), 22050) for i, feat in enumerate(features)]
         
        # # Create a pool of worker processes
        # pool = multiprocessing.Pool(processes=20)

        # # Apply the savemelspec function to each set of parameters in parallel
        # pool.map(savemelspec, params_list)

def extract_save_melspec(audio, fname, folder_path, samplerate):
    
    melspec = librosa.feature.melspectrogram(
                y=audio, sr=samplerate)
       # Convert amplitude to decibels
    mel_spectrogram_db = librosa.power_to_db(melspec, ref=np.max)

    # Plot the Mel spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spectrogram_db, sr=samplerate, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()

    # Specify the folder path to save the images
    #folder_path = 'G:/Cough/Three'
    #folder_path = 'C:/Users/Alliewya/Documents/Cough Monitor/Spectrograms/Denoise_4'
    os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist

    # Save the Mel spectrogram in the folder with the specified filename
    fn = os.path.join(folder_path, fname + "" + ".png")
    plt.savefig(fn)
    plt.close()

def extract_save_mfcc(audio, fname, folder_path, samplerate):
    # Calculate MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=samplerate, n_mfcc=20)

    # Plot the first 13 MFCCs
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs[:13, :], sr=samplerate, x_axis='time')
    plt.colorbar(format='%+2.0f')
    
    # Append title based on fname condition
    if fname.startswith('0'):
        plt.title('MFCCs - Not Cough')
    elif fname.startswith('1'):
        plt.title('MFCCs - Cough')
    else:
        plt.title('MFCCs')
    
    plt.tight_layout()

    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    # Save the MFCC plot as a PNG file
    fn = os.path.join(folder_path, fname + ".png")
    plt.savefig(fn)
    plt.close()

def extract_save_amplitude(audio, fname, folder_path, samplerate, frame_size=2048):
    # Calculate RMS
    rms = librosa.feature.rms(y=audio, frame_length=frame_size, hop_length=frame_size // 2)[0]

    # Plot the amplitude
    plt.figure(figsize=(10, 4))
    frames = range(len(rms))
    t = librosa.frames_to_time(frames, sr=samplerate, hop_length=frame_size // 2)
    plt.plot(t, rms)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    # Append title based on fname condition
    if fname.startswith('0'):
        plt.title('Amplitude - Not Cough')
    elif fname.startswith('1'):
        plt.title('Amplitude - Cough')
    else:
        plt.title('Amplitude')
    #plt.title('Amplitude')
    plt.tight_layout()

    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    # Save the amplitude plot as a PNG file
    fn = os.path.join(folder_path, fname + ".png")
    plt.savefig(fn)
    plt.close()

def plot_mfcc_and_amplitude(audio, fname, folder_path, samplerate):
    # Calculate MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=samplerate, n_mfcc=20)

    # Calculate amplitude (RMS)
    amplitude = librosa.feature.rms(y=audio)

    # Convert frame indices to time values
    frames_mfcc = range(mfccs.shape[1])
    times_mfcc = librosa.frames_to_time(frames_mfcc, sr=samplerate)

    frames_amp = range(amplitude.shape[1])
    times_amp = librosa.frames_to_time(frames_amp, sr=samplerate)

    # Create figure and axes
    fig, ax1 = plt.subplots(figsize=(10, 4))

    # Plot MFCCs
    librosa.display.specshow(mfccs[:13, :], sr=samplerate, x_axis='time', ax=ax1)
    ax1.set(title='MFCCs')

    # Create twin axes for amplitude
    ax2 = ax1.twinx()
    ax2.plot(times_amp, amplitude[0], color='red', alpha=0.5)
    ax2.set_ylabel('Amplitude')

    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    # Save the plot as a PNG file
    fn = os.path.join(folder_path, fname + ".png")
    plt.savefig(fn)
    plt.close()


class MultiPredictor():

    # def returnPredictions(self, filename):

    #     regions = []
    #     filepath = os.path.join("app", "static", "audiofiles",filename)
    #     audio, sr = librosa.load(filepath,sr=22050)
    #     #framelength = sr // 5
    #     framelength = int(sr * 0.2)
    #     hop_length = sr // 100

    #     # Pad up to nearest second
    #     audioduration = librosa.get_duration(y=audio, sr=sr)
    #     roundedduration = math.ceil(audioduration)
    #     paddedaudio = librosa.util.fix_length(audio, size = roundedduration * sr)

    #     # Create frames
    #     frames = librosa.util.frame(
    #         paddedaudio, frame_length=framelength, hop_length=hop_length, axis=0)

    #     # flag for merging overlapping frames
    #     prevframedetected = False

    #     predictions = []
    #     # Predict each frame label
    #     for i, frame in enumerate(frames):

    #         # #old
    #         # features = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=40)
    #         # #print(features.shape)
    #         # features = features.reshape(-1)
    #         # # print(features.shape)
    #         # # a = features.reshape(1,-1)
    #         # # print(a.shape)
    #         # # #print(features.reshape(1,-1))
    #         # # print("_------_----_---_-")
    #         # prediction = classifier.predict(features.reshape(1, -1))
    #         # #print(prediction)
            
            
            
    #         prediction = self.classifier.make_prediction(frame)
    #         predictions.append(prediction[0])
    #         #predictions.append[prediction]
    #         if(prediction[0] == '0'):
                
    #             starttime = i * hop_length / sr
    #             endtime = starttime + (framelength / sr)
    #             if(prevframedetected == True):
    #                 regions[-1]["end"] = endtime
    #             else:
    #                 regions.append({"start": starttime, "end": endtime})
    #             #regions.append({"start": starttime, "end": endtime})
    #             prevframedetected = True
    #         else:
    #             prevframedetected = False
    #         #print(knnmodel.predict(features.reshape(1, -1)))
    #         # print(frame.shape)
            
    #     regions2 = []
    #     regions2 = regions

    #     #remove silent areas
    #     # intervals = librosa.effects.split(paddedaudio)
    #     # print(intervals)
    #     # for i in intervals:
    #     #     print(librosa.samples_to_time(i[0]))
    #     #     print(librosa.samples_to_time(i[1]))
        
    #     ###################
    #     # try:
    #     #     regions = combine_overlapping_times(regions)
    #     # except:
    #     #     print(len(regions))

    #     # for r2 in regions:
    #     #     a = librosa.time_to_samples(r2['start'])
    #     #     b = librosa.time_to_samples(r2['end'])
    #     #     intervals = librosa.effects.split(paddedaudio[a:b])
    #     #     #print(intervals)
    #     #     for i in intervals:
    #     #         regions2.append({"start":librosa.samples_to_time(a + i[0]),"end":librosa.samples_to_time(a + i[1])})

    #     # try:
    #     #     print(type(regions2))
    #     #     print(regions2)
    #     #     regions2 = combine_overlapping_times(regions2)
    #     # except:
    #     #     print(len(regions2))
    #     #     print("Not r2")
    #     #############################################
    #     # print(start_index)
    #     # print(end_index)
    #     # print(frame.shape)
    #     #     print(type(start_index))
    #     #     for i in intervals:
    #     #         print(librosa.samples_to_time(i[0]))
    #     #         print(librosa.samples_to_time(i[1]))
    #     #         regions2.append(
    #     #             {"start":librosa.samples_to_time(i[0]),"end":librosa.samples_to_time(i[1])
    #     #             })
    #     #print(regions2)
    #     #time.sleep(2)

    #     predictiondata = {
    #         "filename": filename,
    #         "regions": regions2
    #     }
    #     #print(predictiondata)
    #     #print(predictions)
    #     return(predictiondata)



    def plot_audio_with_predictions(self, audio, sr, predictions, prediction_probabilities, filename):
        duration = len(audio) / sr
        time_audio = np.linspace(0, duration, len(audio))

        figsizeX = max(20,(duration*4))
        fig, ax1 = plt.subplots(figsize=(30, 12))
        ax1.plot(time_audio, audio, label='Audio waveform')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax2 = ax1.twinx()  # Create a twin axes sharing the x-axis

        #hop_length = int(sr // 10)  
        hop_length = int(sr * 0.2) 
        # Compute the time axis for predictions using librosa.frames_to_time
        frame_start = librosa.frames_to_time(np.arange(len(predictions)), sr=sr, hop_length=hop_length)
        #print(frame_start.shape)
       # frame_start2 =librosa.frames_to_time(frames,sr=sr,hop_length=hop_length)
        #print(frame_start2.shape)

        # frame_duration = int(sr * 0.4)
        # frame_duration = librosa.samples_to_time(frame_duration)
        frame_duration = 0.4
        
        line_height = 0  # Initial line height
        # Plot predicted frames with probabilities
        for probs in prediction_probabilities:
            probs[0][0] = round(probs[0][0],3)
            probs[0][1] = round(probs[0][1],3)


        for i, prediction in enumerate(predictions):
            if i < len(predictions) - 1:
                if prediction == '1':
                    plt.axvspan(frame_start[i], frame_start[i]+frame_duration, color='r', alpha=0.2)
                    plt.text( (frame_start[i] + frame_start[i+1]) / 2,0.25, f"{prediction_probabilities[i]}", ha='center', va='center', rotation=90 )
                    plt.plot([frame_start[i], frame_start[i]+frame_duration], [line_height, line_height], color='lightgray',alpha=0.8)
                    line_height += 0.01  # Decrement line height
                elif prediction == '0':
                    plt.axvspan(frame_start[i], frame_start[i]+frame_duration, color='b', alpha=0.1)
                    plt.text( (frame_start[i] + frame_start[i+1]) / 2,0.25, f"{prediction_probabilities[i]}", ha='center', va='center', rotation=90 )
            else:
                if prediction == '1':
                    plt.axvspan(frame_start[i], frame_duration, color='r', alpha=0.2)
                    plt.text( (frame_start[i] + 0.125) / 2, 0.25, f"Prob{prediction_probabilities[i]}", ha='center', va='center',  rotation=90 )
                    plt.plot([frame_start[i], frame_duration], [line_height, line_height], color='lightgray',alpha=0.8)
                    line_height += 0.01  # Decrement line height
                elif prediction == '0':
                    plt.axvspan(frame_start[i], frame_duration, color='b', alpha=0.1)
                    plt.text((frame_start[i] + 0.125) / 2, 0.25,  f"{prediction_probabilities[i]}", ha='center', va='center',  rotation=90)

        # ax2.plot(frame_start, [probability[0][1] for probability in prediction_probabilities], color='green')

        # Plot the horizontal lines for prediction probabilities
        for i, probability in enumerate(prediction_probabilities):
            ax2.hlines(probability[0][1], frame_start[i], frame_start[i] + 0.2, color='green')
        ax2.set_ylabel('Probability')
        ax2.set_ylim(0, 1) 

        plt.title('Audio with Predicted Frames - ' + filename)
        #plt.legend()

        # Save the plot with the same filename as the input audio
        output_folder = r"G://Cough"
        output_filepath = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.png")
        plt.savefig(output_filepath)
        plt.close()    



    def returnPredictionProbabilities(self, filename):
        regions = []
        filepath = os.path.join("app", "static", "examples", filename)
        #filepath = os.path.join("app","static","examples")
        audio, sr = librosa.load(filepath, sr=22050)
        framelength = int(sr * 0.3)
        #hop_length = int(sr // 10)
        hop_length = int(sr * 0.2)

        audioduration = librosa.get_duration(y=audio, sr=sr)
        roundedduration = math.ceil(audioduration)
        paddedaudio = librosa.util.fix_length(audio, size=roundedduration * sr)

        #30 second segments
        #segments = librosa.util.frame(paddedaudio, frame_length=(sr * 30), hop_length=(sr * 30), axis=0)
        #print(len(segments))
        #for j,segment in enumerate(segments):
            #if j == len(segments) - 1:
            #    break
        frames = librosa.util.frame(
            paddedaudio, frame_length=framelength, hop_length=hop_length, axis=0)
        
        prevframedetected = False

        predictions = []
        prediction_probabilities = []  # Added list for prediction probabilities

        for i, frame in enumerate(frames):
            prediction, probability = self.classifier.make_prediction_probability(frame)  # Obtain prediction and probability
            predictions.append(prediction)
            prediction_probabilities.append(probability)  # Append prediction probability to the list

            if prediction == '0':
                starttime = i * hop_length / sr
                endtime = starttime + (framelength / sr)
                if prevframedetected:
                    regions[-1]["end"] = endtime
                else:
                    regions.append({"start": starttime, "end": endtime})
                prevframedetected = True
            else:
                prevframedetected = False

            regions2 = regions

            predictiondata = {
                "filename": filename,
                "regions": regions2
            }
            #print("Filename:", filename)
            #print("Prediction Probabilities:", prediction_probabilities)  # Print the prediction probabilities
            #fn = filename + "-" + str(j)
        self.plot_audio_with_predictions(paddedaudio, sr, predictions, prediction_probabilities, filename)
        return predictiondata

    def process_audio_files(self, audio_files, classifier):
        with open(r"C:\Users\Alliewya\Documents\Cough Monitor\AudioLabeller\labeller\predictor1.pickle", "rb") as f:
            predictor1 = pickle.load(f)
        self.classifier = predictor1
        print("Reached 3")
        with multiprocessing.Pool(processes=10) as pool:
            results = pool.starmap(self.returnPredictions, [(filename,) for filename in audio_files])

        return results
    
    def process_audio_files_prob(self, audio_files, classifier):
        with open(r"C:\Users\Alliewya\Documents\Cough Monitor\AudioLabeller\labeller\predictor4.pickle", "rb") as f:
            predictor1 = pickle.load(f)
        self.classifier = predictor1
        print("Reached 3")
        with multiprocessing.Pool(processes=10) as pool:
            results = pool.starmap(self.returnPredictionProbabilities, [(filename,) for filename in audio_files])

        return results
    

def dataset_stats_calculate(datasetfname):
    path = os.path.join("app", "static", "datasetpickles", datasetfname)
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
        stats = {}
        total_length = 0
        for audio in dataset.samples:
            total_length = total_length + librosa.get_duration(y=audio.audiodata)
        count = len(dataset.samples)
        average = total_length/count
        stats["Total Length"] = total_length
        stats["Average Length"] = average
        stats["Audio"] = count
        int_list = [int(item) for item in dataset.labels]
        stats["Label 1"] = sum(int_list)
        stats["Label 0"] = len(dataset.labels)- stats["Label 1"]
        stats["Sample Rate"] = dataset.samples[0].samplerate
        filesizebytes =  os.path.getsize(path)
        filesizegb = filesizebytes / (1024 ** 3)
        stats["Filesize"] = f"{filesizegb:.2f}"
        
    del dataset
    print(str(datasetfname) + str(stats))
    return stats