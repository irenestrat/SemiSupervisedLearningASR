from absl import app
from absl import flags
from pathlib import Path
import pandas as pd
import os
from scipy.io import wavfile
import scipy.io
import subprocess
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets.data_loaders_2 import TimitDataset
from datasets.data_transformations import MFCC
from datasets.data_transformations import Phonemes
from datasets.corpus import *

FLAGS = flags.FLAGS

def main(argv):
    """ This is our main method.
    """
    del argv  # Unused.
    
    # Initialize a Corpus object
    example_file_dir = "/data/TRAIN/DR1/FCJF0/SA1"  #SA1.wav.WAV
    dataset_dir = "/home/georgmosh/Documents/SpeechLabs/dt2119_semisup_project/SemiSupervisedLearningASR-main/timit"
    # dataset_dir = '../timit'
    corpus = Corpus(dataset_dir, example_file_dir) # TIMIT corpus
    phonemes = corpus.get_phonemes()  # List of phonemes
    targets = len(phonemes)  # Number of categories

    # Load the TIMIT dataset
    dataset = TimitDataset(csv_file = 'test_data.csv',
                           root_dir = dataset_dir,
                           corpus = corpus,
                           transform = MFCC(n_fft=FLAGS.n_fft,
                                          preemphasis_coefficient=FLAGS.preemphasis_coefficient,
                                          num_ceps=FLAGS.num_ceps),
                           transcription = Phonemes(n_fft=FLAGS.n_fft,
                                           preemphasis_coefficient=FLAGS.preemphasis_coefficient,
                                           num_ceps=FLAGS.num_ceps,
                                           corpus = corpus))

    ele = dataset[1]
#    print(ele['phonemes_per_frame'])
    # Get the MFCC coefficients
    train_data, max_len = getMFCCFeatures(dataset)
    
    # Get the phonemes per frame (as percentages)
    train_targets = getTargetPhonemes(dataset, max_len, corpus, train_data)

    
def getMFCCFeatures(dataset, zeropad = False):
    """ This method computes the MFCC coefficients per frame.
         When frames are less than the maximum amount does zero-padding.
         @returns tensors of MFCC coefficients of the same length
    """
    features = []
    tensors = []
    max_frames = -1

    for i in range(len(dataset)):
            sample = dataset[i]
            audio = np.asarray(sample['audio'])
            if(max_frames < audio.shape[0]):
                max_frames = audio.shape[0]
            features.append(audio)
    
    if(zeropad == True):
        # zero-padding for equal length
        for i in range(len(dataset)):
            audio_new = np.zeros((max_frames, features[i].shape[1]))
            audio_new[0:features[i].shape[0],:] = features[i]
            tensors.append(torch.tensor(audio_new.tolist(), dtype=torch.long))
    else:
        for i in range(len(dataset)):
            audio_new = features[i]
            tensors.append(torch.tensor(audio_new.tolist(), dtype=torch.long))

    return tensors, max_frames

def getTargetPhonemes(dataset, max_frames, corpus, mfccs, zeropad = False, onehot = True):
    """ This method computes the target phonemes as percentages per frame.
         @returns tensors of phonemes per frame
    """
    tensors = []
    targets = []
    max_frames = -1

    for i in range(len(dataset)):
        sample = dataset[i]
        phoneme_list = sample['phonemes_per_frame']
        sample_targets = []
        for j in range(len(phoneme_list)):
            if(onehot == False):
                if(len(phoneme_list[j]) == 1):
                    # only one phoneme in this frame --> one hot encoding
                    the_one_phoneme = phoneme_list[j][0][2]
                    the_one_phoneme_one_hot = corpus.phones_to_onehot([the_one_phoneme])[0]
                    sample_targets.append(the_one_phoneme_one_hot)
                else:
                    # more than one phonemes in this frame --> probabilities
                    complex_one_hot = np.zeros(len(corpus.get_phonemes())).tolist()
                    for k in range(len(phoneme_list[j])):
                        the_kth_phoneme = phoneme_list[j][k][2]
                        the_kth_phoneme_ID = corpus.get_phones_ID(the_kth_phoneme)
                        the_kth_phoneme_percentage = phoneme_list[j][k][3]
                        complex_one_hot[the_kth_phoneme_ID] = the_kth_phoneme_percentage
                        sample_targets.append(complex_one_hot)
            else:
                # using only one phoneme explicitly imposed --> one hot encoding
                the_one_phoneme = phoneme_list[j][0][2]
                the_one_phoneme_one_hot = corpus.phones_to_onehot([the_one_phoneme])[0]
                sample_targets.append(the_one_phoneme_one_hot)
        targets.append(sample_targets)
    
    if(zeropad == True):
        # silence-padding for equal length (like zero-padding for the MFCCs - don't know if works)
        for i in range(len(dataset)):
            sample_targets = targets[i]
            for j in range(len(sample_targets), max_frames, 1):
                # adding a silence target for the extra frames (zero-padded additions)
                silence = corpus.get_silence()
                silence_one_hot = corpus.phones_to_onehot([silence])[0]
                targets[i].append(silence_one_hot)
        
    for i in range(len(targets)):
        tensors.append(torch.tensor(targets[i], dtype=torch.long))

    return tensors

if __name__ == '__main__':
    flags.DEFINE_integer('n_fft', 512, 'Size of FFT')
    flags.DEFINE_float('preemphasis_coefficient', 0.97,
                       'Coefficient for use in signal preemphasis')
    flags.DEFINE_integer(
        'num_ceps', 13, ' Number of cepstra in MFCC computation')

    app.run(main)
