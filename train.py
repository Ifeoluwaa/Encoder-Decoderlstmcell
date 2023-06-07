import random
import pandas as pd

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import nltk
from nltk.corpus import stopwords
from nltk.translate.bleu_score import sentence_bleu
from torchtext.data.metrics import bleu_score
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from process_data import  tokenize, word_frequency, word_to_index_map, idx_to_word, create_vocab, train_test_split, tensorFromSentence 
from EncoderDecoder import  Encoderlstm, Decoderlstm, lstm_seq2seq
MAXIMUM_LENGTH = 30

SOS_token = 0
EOS_token = 1
criterion = nn.NLLLoss()
step = 1

stop_words = set(stopwords.words("english"))

#Checking if gpu is installed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#load the data file
#The file can be read using the file name as string or an open file object:
file_path = 'eng_yor_data.xlsx'
data_file = pd.read_excel(file_path, index_col='ID')

dataset = [tuple(r) for r in data_file.to_numpy().tolist()]
#print(dataset[0])
 
data_set = [[tokenize(x[0]), x[1].split()] for x in dataset]
#print(data_set)


#Spliting the dataset   
train_data, test_data = train_test_split(data_set, train_size=0.75)
#print(len(train_data))


#Creating vocabulary for the train_data
english_vocabulary, yoruba_vocabulary = create_vocab(train_data)
#print(len(english_vocabulary))


#Mapping of numbers to english words
english_word_map = word_to_index_map(english_vocabulary)
#print(len(english_word_map))


#Mapping numbers to yoruba words
yoruba_word_map = word_to_index_map(yoruba_vocabulary)
#print(len(yoruba_word_map))


#A = torch.tensor(word_to_index_map[x.items()] for x in data_set)

#print(tuple(torch.from_numpy(item).to(device=device, dtype=torch.float32) for item in A))
wordTensors = tensorFromSentence(train_data[0][0], english_word_map)
#print(wordTensors)


#For encoder(Parameters)
input_size_encoder = len(english_word_map)
encoder_embedding_size = 260
hidden_size = 526


Encoder_LSTM = Encoderlstm(input_size_encoder, encoder_embedding_size, hidden_size).to(device)
print(Encoder_LSTM)


#For Decoder(Parameters)
encoder_embeddiing_size = 260
decoder_embedding_size = 260
hidden_size = 526
output_size = len(yoruba_word_map)

Decoder_LSTM = Decoderlstm(output_size, decoder_embedding_size, hidden_size).to(device)
print(Decoder_LSTM)

EPOCHS = 5
#Training the model
model = lstm_seq2seq(input_size_encoder, output_size, encoder_embedding_size, decoder_embedding_size, hidden_size)
#print(model)

optimizer = torch.optim.Adam(model.parameters())

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#print(f'The model has {count_parameters(model):,} trainable parameters')


optimizer = torch.optim.Adam(model.parameters())

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#print(f'The model has {count_parameters(model):,} trainable parameters')

#model.train(True)

for epoch in range(EPOCHS):
    for i in train_data:

      model.zero_grad()

      english_tensors = tensorFromSentence(i[0], english_word_map)
      yoruba_tensors = tensorFromSentence(i[1], yoruba_word_map)
      output = model.forward(english_tensors, yoruba_tensors).to(device)
      #print(len(output))

      loss = criterion(output, torch.reshape(yoruba_tensors, (yoruba_tensors.shape[0],)))

      loss.backward()

      optimizer.step()

      print(f"At Epoch {epoch}, at iteration {step}, loss: {loss}")
      step += 1
    
    
    
    
#Testing
score = 0
count_scores = 0
for pairs in test_data:
  with torch.no_grad():
    test_source = tensorFromSentence(pairs[0], english_word_map)
    test_target = tensorFromSentence(pairs[1], yoruba_word_map)

    predicted_sentence = []
    output = model.forward(test_source, test_target)
    _, predictions = output.topk(k=1)
    for prediction in predictions:
      predicted_sentence.append(idx_to_word(prediction.item(), yoruba_word_map))

    print("Source: ", pairs[0])
    print("Target: ", pairs[1])
    print("Prediction: ", predicted_sentence)
    predicted_sentence.pop()
    score += sentence_bleu([pairs[1]], predicted_sentence, weights=(1, 0, 0, 0))
    count_scores += 1
    print()

print("bleu_score:", score/count_scores * 100)
#With Epoch=5, bleu_score = 52.49