import os
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as transforms

import wer

'''Encode text and decode numeric output from model (one hot)'''
def encode_text(text):
    labels = [ord(letter) for letter in text]
    return labels

def decode_text(label):
    char = [chr(num) for num in label]
    text = ''.join([str(elem) for elem in char])
    return text

train_transforms = nn.Sequential(
    transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
    transforms.FrequencyMasking(15),
    transforms.TimeMasking(35)
)
test_transforms = transforms.MelSpectrogram(sample_rate=16000, n_mels=128)

'''Extracts waveform and transcript from tuple of data for each audio sample
Transforms into spectrogram and calculates 
length of waveform and transcript for use in CTC loss'''
def process_data(data, data_type):
    size = len(data)
    spectrograms = [0]*size
    labels = [0]*size
    input_len = [0]*size
    label_len = [0]*size
    idx = 0
    
    for item in data:
        if data_type == 'train':
            spec = train_transforms(item[0])
        else:
            spec = test_transforms(item[0])
        spectrograms[idx] = spec.transpose(0,2).squeeze()  # Make variable dim the singleton dim for padding later
        label = torch.Tensor(encode_text(item[2].lower()))
        labels[idx] = label
        input_len[idx] = len(spectrograms)//2  # Calculate length before padding
        label_len[idx] = len(label)
        idx += 1
        
    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    
    return spectrograms, labels, input_len, label_len

def accuracy(output, transcript):
    encoded_chars = torch.argmax(output, dim=2)
    encoded_chars = encoded_chars[encoded_chars!=0]  # Remove blanks
    prediction = decode_text(encoded_chars)
    return wer.wer(prediction, transcript)

'''Loads audio data and transcripts
Call load_l2arctic() or load_librispeech separately
To train with both datasets, combine train/test datasets
then run process_data() on combined arrays'''
class input_data():

    def load_l2arctic(self):

        def load_audio():
            audio = [0]*17*1131
#           audio = [0]*17
            idx=0

            for speaker in range(17):
                files = os.listdir('../l2arctic_recordings/'+str(speaker))
#               individual_audio = [0]*1131
                for file in files:
                    waveform, sample_rate = torchaudio.load(file)
                    
                    # Match sampling rate between corpuses
                    resample = torchaudio.transforms.Resample(sample_rate, 16000)
                    waveform = resample(waveform)

#                    individual_audio[idx] = waveform
                    audio[idx] = waveform
                    idx+=1
#                    audio[speaker] = individual_audio
            return audio
    
        audio = load_audio()
        transcript = np.loadtxt('../arctic_transcript.txt', 'r')

        # Separate into training/testing data sets
        train_data = [0]*904
        test_data = [0]*225
        train_idx = 0
        test_idx = 0
        i = 0
        for item in (audio, transcript):
            if (i%5 == 0):
                test_data[test_idx] = item
                test_idx+=1
            else:
                train_data[train_idx] = item
                train_idx+=1
            i+=1
        
        return train_data, test_data

    def load_librispeech(self):
        train_data = torchaudio.datasets.LIBRISPEECH('./', url='train-clean-100', download=True)
        test_data = torchaudio.datasets.LIBRISPEECH('./', url='test-clean', download=True)

        return train_data, test_data