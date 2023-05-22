import numpy as np
from kymatio.numpy import Scattering1D
import librosa

def load_audio_file(file_path, sample_rate=22050):
    audio_data, _ = librosa.load(file_path, sr=sample_rate, mono=True)
    return audio_data

def extract_features(audio_data, sample_rate=22050, J=6, Q=1):
    T = len(audio_data)
    scattering = Scattering1D(J, T, Q)
    features = scattering(audio_data)
    return features

def average_features(features, axis=1):
    avg_features = np.mean(features, axis=axis)
    return avg_features

if __name__ == '__main__':
    file_path = r'C:\Users\Alliewya\Documents\Cough Monitor\AudioLabeller\labeller\app\audiofiles\fc6a34fe-7f14-42e1-8483-d1a8f68a8d88.wav'
    
    sample_rate = 22050
    J = 8  # The maximum scale of the scattering transform (2**J should be smaller than the signal length)
    Q = 1  # The number of wavelets per octave
    
    audio_data = load_audio_file(file_path, sample_rate)
    features = extract_features(audio_data, sample_rate, J, Q)
    print("Feature shape:", features.shape)
    avg_features = average_features(features)
    
    print("Averaged feature shape:", avg_features.shape)