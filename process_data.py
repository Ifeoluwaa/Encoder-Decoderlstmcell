import string
import re
import random
import pandas as pd

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from itertools import islice


stop_words = set(stopwords.words("english"))

#Checking if gpu is installed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAXIMUM_LENGTH = 30
SOS_token = 0
EOS_token = 1
#load the data file

file_path = 'eng_yor_data.xlsx'
data_file = pd.read_excel(file_path, index_col='ID')

dataset = [tuple(r) for r in data_file.to_numpy().tolist()]
#print(dataset[0])


#Tokenizing
def tokenize(sentence):
    # Create tokenizer  
    listofsentence = word_tokenize(sentence)
    return listofsentence

#frequency of words in the above tokenization
def word_frequency(data):
    words_term_frequency_train = {}
    for line in data:
        for word in line:
            if word not in words_term_frequency_train:
                words_term_frequency_train[word] = 1
            else:
                words_term_frequency_train[word] = words_term_frequency_train.get(word, 0) + 1
    return words_term_frequency_train


#Mapping of words to numbers
def word_to_index_map(vocabulary):
    word_to_ix = {0: "SOS", 1: "EOS"}
    for i, word in enumerate(vocabulary):
        word_to_ix[word] = i + 2
    return word_to_ix

def idx_to_word(idx, word_to_index_map):
    for key, value in word_to_index_map.items():
        if value == idx:
            return key

#Create vocabulary
def create_vocab(corpus):
    english_vocab = []
    yoruba_vocab = []
    for engword, yorword in corpus:
        for word in engword:
            if word not in english_vocab:
                english_vocab.append(word)
        for word in yorword:
            if word not in yoruba_vocab:
                yoruba_vocab.append(word)
    return english_vocab, yoruba_vocab


#Split the dataset
def train_test_split(corpus, train_size, shuffle=False):
    if shuffle:
        random.shuffle(corpus)
    train_size = int(train_size * len(corpus))
    train_arr = corpus[:train_size]
    test_arr = corpus[train_size + 1:]
    return train_arr, test_arr


def filterpair(p):
    filtered_words = []
    for english, yoruba in p:
        if len(english) < MAXIMUM_LENGTH and len(yoruba) < MAXIMUM_LENGTH:
            filtered_words.append([english, yoruba])
    return filtered_words 


def tensorFromSentence(sentence, word_mapping):
    indexes = [word_mapping[word] for word in sentence if word in word_mapping]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


#def tensorsFromPair(english_vocab, yoruba_vocab):
 #   input_tensor = tensorFromSentence(english_vocab, filterpair[0])
  #  target_tensor = tensorFromSentence(yoruba_vocab, filterpair[1])
  #  return (input_tensor, target_tensor)
