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
    dataset = TimitDataset(csv_file = 'train_data.csv',
                           root_dir = dataset_dir,
                           corpus = corpus,
                           transform = MFCC(n_fft=FLAGS.n_fft,
                                          preemphasis_coefficient=FLAGS.preemphasis_coefficient,
                                          num_ceps=FLAGS.num_ceps),
                           transcription = Phonemes(n_fft=FLAGS.n_fft,
                                           preemphasis_coefficient=FLAGS.preemphasis_coefficient,
                                           num_ceps=FLAGS.num_ceps,
                                           corpus = corpus))
    
    test_dataset = TimitDataset(csv_file = 'test_data.csv',
                           root_dir = dataset_dir,
                           corpus = corpus,
                           transform = MFCC(n_fft=FLAGS.n_fft,
                                          preemphasis_coefficient=FLAGS.preemphasis_coefficient,
                                          num_ceps=FLAGS.num_ceps),
                           transcription = Phonemes(n_fft=FLAGS.n_fft,
                                           preemphasis_coefficient=FLAGS.preemphasis_coefficient,
                                           num_ceps=FLAGS.num_ceps,
                                           corpus = corpus))

    # Get the MFCC coefficients
    train_data, max_len = getMFCCFeatures(test_dataset, oneTensor = True)
    print(train_data.shape)
    
    # Get the phonemes per frame (as percentages)
    train_targets = getTargetPhonemes(test_dataset, max_len, corpus, oneTensor = True, mode = "percentages")
    print(train_targets.shape)
    
    
def getMFCCFeatures(dataset, zeropad = False, oneTensor = False):
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
            if(zeropad == True):
                features.append(audio)
            else:
                tensors.append(torch.tensor(audio.tolist(), dtype=torch.long))
    
    if(zeropad == True):
        # zero-padding for equal length
        for i in range(len(dataset)):
            audio_new = np.zeros((max_frames, features[i].shape[1]))
            audio_new[0:features[i].shape[0],:] = features[i]
            tensors.append(torch.tensor(audio_new.tolist(), dtype=torch.long))
    
    if(oneTensor == True):
        whole = tensors[0].numpy()
        for i in range(1, len(dataset)):
            whole = np.concatenate((whole, tensors[i].numpy()), axis = 0)
        tensors = torch.tensor(whole.tolist(), dtype = torch.long)

    return tensors, max_frames

def getTargetPhonemes(dataset, max_frames, corpus, zeropad = False, oneTensor = False, mode = "indices"):
    """ This method computes the target phonemes as percentages per frame.
         @returns tensors of phonemes per frame
    """
    tensors = []
    targets = []

    for i in range(len(dataset)):
        sample = dataset[i]
        phoneme_list = sample['phonemes_per_frame']
        sample_targets = []
        
        for j in range(len(phoneme_list)):
            if(mode == "indices"):
                # using only one phoneme explicitly imposed --> first phoneme index
                the_one_phoneme = phoneme_list[j][0][2]
                the_one_phoneme_id = corpus.get_phones_ID(the_one_phoneme)
                sample_targets.append(the_one_phoneme_id)
            elif(mode == "percentages"):
                if(len(phoneme_list[j]) == 1):
                    # only one phoneme in this frame --> one hot encoding
                    the_one_phoneme = phoneme_list[j][0][2]
                    the_one_phoneme_one_hot = corpus.phones_to_onehot([the_one_phoneme])[0]
                    sample_targets.append(the_one_phoneme_one_hot)
                else:
                    # more than one phonemes in this frame --> probabilities
                    complex_one_hot = np.zeros(corpus.get_phones_len()).tolist()
                    for k in range(len(phoneme_list[j])):
                        the_kth_phoneme = phoneme_list[j][k][2]
                        the_kth_phoneme_ID = corpus.get_phones_ID(the_kth_phoneme)
                        the_kth_phoneme_percentage = phoneme_list[j][k][3]
                        complex_one_hot[the_kth_phoneme_ID] = the_kth_phoneme_percentage
                    sample_targets.append(complex_one_hot)
            elif(mode == "onehot"):
                # using only one phoneme explicitly imposed --> one hot encoding
                the_one_phoneme = phoneme_list[j][0][2]
                the_one_phoneme_id = corpus.get_phones_ID(the_one_phoneme)
                the_one_phoneme_one_hot = corpus.phones_to_onehot([the_one_phoneme])[0]
                sample_targets.append(the_one_phoneme_one_hot)
            else:
                print("Wrong mode!")
                break
        
        if(zeropad == True):
            for j in range(len(phoneme_list), max_frames):
                # adding a silence target for the extra frames (zero-padded additions)
                silence = corpus.get_silence()
                if(mode == "percentages" or mode == "onehot"):
                    silence_one_hot = corpus.phones_to_onehot([silence])[0]
                    sample_targets.append(silence_one_hot)
                elif(mode == "indices"):
                    silence_id = corpus.get_phones_ID(silence)
                    sample_targets.append(silence_id)
                
        tensors.append(torch.tensor(sample_targets, dtype=torch.long))
        
    if(oneTensor == True):
        whole = tensors[0].numpy()
        for i in range(1, len(dataset)):
            whole = np.concatenate((whole, tensors[i].numpy()), axis = 0)
        tensors = torch.tensor(whole.tolist(), dtype = torch.long)

    return tensors

if __name__ == '__main__':
    flags.DEFINE_integer('n_fft', 512, 'Size of FFT')
    flags.DEFINE_float('preemphasis_coefficient', 0.97,
                       'Coefficient for use in signal preemphasis')
    flags.DEFINE_integer(
        'num_ceps', 13, ' Number of cepstra in MFCC computation')

    app.run(main)
