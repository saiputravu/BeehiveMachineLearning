import numpy as np
import pandas as pd
import librosa
import tqdm
import ast
import matplotlib.pyplot as plt

NFFT = 512
HOPLENGTH = NFFT//4
NMELS = 128
FIGSIZE = (18,9)

DATADIR = '../data/sound_files/'
IMAGEDIR = '../data/image_files/'

def spectrogram(signal, sr, n_fft=NFFT, hop_length=HOPLENGTH):
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    spec = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    return spec

def melspectrogram(signal, sr, n_mels=NMELS, **kwargs):
    spec = spectrogram(signal, sr, **kwargs)
    mel = librosa.feature.melspectrogram(S=spec, sr=sr, n_mels=n_mels)
    return mel
    
def f_of_file(file, f, **kwargs):
    signal, sr = librosa.load(file)
    return f(signal, sr, **kwargs), sr

def mel_of_file(file, **kwargs):
    return f_of_file(file, melspectrogram)

def display_spec(spec, sr, y_axis='mel', figsize=FIGSIZE, n_fft=NFFT, hop_length=HOPLENGTH, **kwargs):
    # Display spectogram
    plt.figure(figsize=FIGSIZE)
    librosa.display.specshow(spec, y_axis=y_axis, x_axis='time', 
                             sr=sr, n_fft=n_fft, 
                             hop_length=hop_length, **kwargs)

    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar(format="%+2.f dB")
    plt.title("Spectrogram")
    plt.show()

def dump_mel(mel, sr, f, **kwargs):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(10,10)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    im = librosa.display.specshow(mel, y_axis='mel', x_axis='time', 
                                  sr=sr, n_fft=NFFT, 
                                  hop_length=HOPLENGTH, ax=ax,
                                  **kwargs)
    fig.savefig(f, dpi=160)
    plt.close()

meta = pd.read_csv('../data/data.csv')
meta['segments'] = meta['segments'].apply(ast.literal_eval)

print("Starting")
for indx,value in tqdm.tqdm(meta[["file name", "segments"]].iterrows()):
    f, segments = value.to_numpy()
    for s in tqdm.tqdm(segments, leave=False):
        wav = f"{DATADIR}/{f}__segment{s}.wav"
        img = f"{IMAGEDIR}/{f}__segment{s}.png"
        if os.path.exists(img):
            # Skip over already generated images. (assuming nothing is corrupted)
            continue
        mel, sr = mel_of_file(wav)
        dump_mel(mel, sr, f=img)
