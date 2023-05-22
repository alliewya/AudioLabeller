import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import pywt
import os
import shutil
import librosa
import librosa.display
from concurrent.futures import ProcessPoolExecutor
from scipy.signal import cwt, ricker
import networkx as nx
#import pygraphviz as pgv
from networkx.drawing.nx_agraph import graphviz_layout

# Increase Chunk size for plt
plt.rcParams['agg.path.chunksize'] = 10000


def plot_waveform_and_spectrogram(data, rate, filename, wavelet):
    plt.figure(figsize=(20, 12), dpi=400)

    plt.subplot(3, 1, 1)
    plt.plot(data)
    plt.title(f'Waveform ({wavelet})')
    plt.xlim(0, len(data))

    plt.subplot(3, 1, 2)
    plt.specgram(data, NFFT=2048, Fs=rate, noverlap=1900, cmap='plasma')
    plt.title(f'Spectrogram ({wavelet})')

    plt.subplot(3, 1, 3)
    data_float = data.astype(np.float32) / np.iinfo(data.dtype).max
    S = librosa.feature.melspectrogram(
        y=data_float, sr=rate, n_fft=2048, hop_length=1024, n_mels=128)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr=rate,
                             hop_length=1024, y_axis='mel', fmax=8000, x_axis='time', cmap='plasma')
    plt.title(f'Mel Spectrogram ({wavelet})')
    #plt.colorbar(format='%+2.0f dB')

     # Scalogram
    # plt.subplot(4, 1, 4)
    # widths = np.arange(1, 101)
    # cwtmatr = cwt(data_float, ricker, widths)
    # plt.imshow(cwtmatr, extent=[0, len(data)/rate, 1, 101], cmap='plasma', aspect='auto',
    #            vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
    # plt.title(f'Scalogram ({wavelet})')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Scales (Widths)')
    # plt.colorbar(label='Magnitude')

    plt.tight_layout()
    plt.savefig(filename, dpi=400)
    plt.close()

def plot_wavelet_coeffs(coeffs, filename):
    fig, axes = plt.subplots(len(coeffs), 1, figsize=(10, 2 * len(coeffs)), dpi=300, sharex=True, sharey=True)
    for i in reversed(range(len(coeffs))):
        ax = axes[i]
        if i == 0:
            ax.set_title(f'Approximation Coefficients at Level {len(coeffs)-i}')
        else:
            ax.set_title(f'Detail Coefficients at Level {len(coeffs)-i}')
        ax.plot(coeffs[i])
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()





def plot_all_waveforms(inputs, outputs, rate, filename):
    plt.figure(figsize=(20, 12), dpi=400)

    for i, data in enumerate(inputs + outputs):
        plt.subplot(len(inputs + outputs), 1, i + 1)
        plt.plot(data)
        if i == 0:
            plt.title('Input Waveform')
        else:
            plt.title(f'Subband {i} Waveform')

    plt.tight_layout()
    plt.savefig(filename, dpi=400)
    plt.close()


def plot_all_spectrograms(inputs, outputs, rate, filename):
    plt.figure(figsize=(20, 20), dpi=400)

    for i, data in enumerate(inputs + outputs):
        plt.subplot(len(inputs + outputs), 1, i + 1)
        plt.specgram(data, NFFT=2048, Fs=rate, noverlap=1900,  cmap='plasma')
        if i == 0:
            plt.title('Input Spectrogram')
        else:
            plt.title(f'Subband {i} Spectrogram')

    plt.tight_layout()
    plt.savefig(filename, dpi=400)
    plt.close()


def plot_all_mel_spectrograms(inputs, outputs, rate, filename):
    plt.figure(figsize=(20, 20), dpi=400)

    for i, data in enumerate(inputs + outputs):
        plt.subplot(len(inputs + outputs), 1, i + 1)
        data_float = data.astype(np.float32) / np.iinfo(data.dtype).max
        S = librosa.feature.melspectrogram(
            y=data_float, sr=rate, n_fft=2048, hop_length=1024, n_mels=128)
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr=rate,
                                 hop_length=1024, y_axis='mel', fmax=8000, x_axis='time', cmap='plasma')
        if i == 0:
            plt.title('Input Mel Spectrogram')
        else:
            plt.title(f'Subband {i} Mel Spectrogram')

    plt.tight_layout()
    plt.savefig(filename, dpi=400)
    plt.close()

def plot_all_scalograms(inputs, outputs, rate, filename, wavelet):
    plt.figure(figsize=(20, 20), dpi=400)

    for i, data in enumerate(inputs + outputs):
        data_float = data.astype(np.float32) / np.iinfo(data.dtype).max

        plt.subplot(len(inputs + outputs), 1, i + 1)
        widths = np.arange(1, 101)
        cwtmatr = cwt(data_float, ricker, widths)
        plt.imshow(cwtmatr, extent=[-1, 1, 1, 101], cmap='plasma', aspect='auto',
                   vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
        if i == 0:
            plt.title('Input Scalogram')
        else:
            plt.title(f'Subband {i} Scalogram')
        plt.xlabel('Time (s)')
        plt.ylabel('Scales (Widths)')

    plt.tight_layout()
    plt.savefig(filename, dpi=400)
    plt.close()

def plot_signal_energy(subbands, filename):
    num_subbands = len(subbands)
    fig, axs = plt.subplots(num_subbands, 1, figsize=(10, 5 * num_subbands))
    for i, subband in enumerate(subbands):
        energy = np.abs(subband) ** 2
        axs[i].plot(energy)
        axs[i].set_title(f'Signal Energy Over Time for Subband {i}')
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel('Energy')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# def plot_wavelet_tree(coeffs, filename):
#     G = nx.DiGraph()
#     G.add_node('Signal')
#     for i, coeff in enumerate(coeffs):
#         for j in range(len(coeff)):
#             G.add_node(f'Level {i}, Coeff {j}')
#             G.add_edge('Signal' if i == 0 else f'Level {i-1}, Coeff {j//2}', f'Level {i}, Coeff {j}')
#     pos = graphviz_layout(G, prog='dot')
#     plt.figure(figsize=(10, 10))
#     nx.draw(G, pos, with_labels=True, arrows=False)
#     plt.savefig(filename)
#     plt.close()

# def plot_wavelet_tree_matplot(coeffs, filename):
#     G = nx.DiGraph()
#     G.add_node('Signal')
#     for i, coeff in enumerate(coeffs):
#         for j in range(len(coeff)):
#             G.add_node(f'Level {i}, Coeff {j}')
#             G.add_edge('Signal' if i == 0 else f'Level {i-1}, Coeff {j//2}', f'Level {i}, Coeff {j}')
#     pos = nx.spring_layout(G)  # Use spring layout instead
#     plt.figure(figsize=(10, 10))
#     nx.draw(G, pos, with_labels=True, arrows=False)
#     plt.savefig(filename)
#     plt.close()

def plot_subband_tree_matplot(subbands, filename):
    G = nx.DiGraph()
    G.add_node('Signal')
    for i, subband in enumerate(subbands):
        G.add_node(f'Subband {i}')
        G.add_edge('Signal', f'Subband {i}')
    pos = nx.spring_layout(G)  # Use spring layout instead
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, with_labels=True, arrows=False)
    plt.savefig(filename)
    plt.close()


def process_audio_file(input_file, wavelet, level=3):
    rate, data = wavfile.read(input_file)

    if data.ndim > 1:
        data = data[:, 0]

    coeffs = pywt.wavedec(data, wavelet, level=level)

    subbands = []
    for i in range(len(coeffs)):
        zero_coeffs = [np.zeros_like(c) for c in coeffs]
        zero_coeffs[i] = coeffs[i]
        subband = pywt.waverec(zero_coeffs, wavelet)
        subbands.append(subband.astype(data.dtype))

    wavelet_folder = wavelet
    if not os.path.exists(wavelet_folder):
        os.makedirs(wavelet_folder)

    level_folder = os.path.join(wavelet_folder, f"level_{level}")
    if not os.path.exists(level_folder):
        os.makedirs(level_folder)

    file_basename = os.path.splitext(os.path.basename(input_file))[0]
    output_folder = os.path.join(level_folder, file_basename)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Copy input .wav file to output folder
    shutil.copy(input_file, output_folder)

    for i, subband in enumerate(subbands):
        name = f"Subband_{i}"
        wavfile.write(os.path.join(
            output_folder, f'{name}.wav'), rate, subband)
        plot_waveform_and_spectrogram(subband, rate, os.path.join(
            output_folder, f'{name}.png'), wavelet)

    plot_waveform_and_spectrogram(data, rate, os.path.join(
        output_folder, 'input.png'), wavelet)
    plot_all_waveforms([data], subbands, rate, os.path.join(
        output_folder, 'all_waveforms.png'))
    plot_all_spectrograms([data], subbands, rate, os.path.join(
        output_folder, 'all_spectrograms.png'))
    plot_all_mel_spectrograms([data], subbands, rate, os.path.join(
        output_folder, 'all_mel_spectrograms.png'))
    #plot_all_scalograms([data], subbands, rate, os.path.join(
    #    output_folder, 'all_scalograms.png'), wavelet)

    # Plot wavelet coefficients
    plot_wavelet_coeffs(coeffs, os.path.join(output_folder, 'wavelet_coeffs.png'))
    

    # Plot signal energy
    plot_signal_energy(subbands, os.path.join(output_folder, 'signal_energy.png'))

    # Plot wavelet tree
    #plot_wavelet_tree(coeffs, os.path.join(output_folder, 'wavelet_tree.png'))

    # Plot Wavelet Using Matplot
    plot_subband_tree_matplot(subbands, os.path.join(output_folder, 'wavelet_tree_matplot.png'))


if __name__ == '__main__':
    # Possible wavelets: 'haar', 'db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'coif1', 'coif2', 'coif3', 'coif4', 'coif5', 'sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7', 'sym8'
    wavelet = 'db4'

    max_processes = 8

    source_folder = 'source'
    input_files = [os.path.join(source_folder, file) for file in os.listdir(
        source_folder) if file.endswith('.wav')]

    with ProcessPoolExecutor(max_workers=max_processes) as executor:
        for _ in executor.map(process_audio_file, input_files, [wavelet] * len(input_files)):
            pass

    wavelet = "haar"

    with ProcessPoolExecutor(max_workers=max_processes) as executor:
        for _ in executor.map(process_audio_file, input_files, [wavelet] * len(input_files)):
            pass

    wavelet = "coif5"

    with ProcessPoolExecutor(max_workers=max_processes) as executor:
        for _ in executor.map(process_audio_file, input_files, [wavelet] * len(input_files)):
            pass

    wavelet = "sym8"

    with ProcessPoolExecutor(max_workers=max_processes) as executor:
        for _ in executor.map(process_audio_file, input_files, [wavelet] * len(input_files)):
            pass
