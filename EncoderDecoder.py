import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

#Creating class encoder using LSTM Network

class Encoderlstm(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim):
        super(Encoderlstm, self).__init__()
        
        self.hid_dim = hid_dim
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.lstmcell = nn.LSTMCell(input_size=emb_dim, hidden_size=hid_dim)
        
        
    def forward(self, input_tensor, hidden_state, cell_state):
        
        embedded = self.embedding(input_tensor)
        hidden_state, cell_state = self.lstmcell(embedded, (hidden_state, cell_state))
        return hidden_state, cell_state
      
        
    def init_hidden(self):
        hidden_state = torch.zeros((1, self.hid_dim), device=device)
        cell_state = torch.zeros((1, self.hid_dim), device=device)
        
        torch.nn.init.xavier_normal_(hidden_state)
        torch.nn.init.xavier_normal_(cell_state)
        
        return hidden_state, cell_state

#Creating class Decoder
class Decoderlstm(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim):
        super(Decoderlstm, self).__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.lstmcell = nn.LSTMCell(input_size=emb_dim + hid_dim, hidden_size=hid_dim)
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        
    def forward(self, input_tensor, context_tensor, hidden_state, cell_state):
        
        embedded = self.embedding(input_tensor)
        context_and_embed = torch.cat((embedded, context_tensor), -1)
        hidden_state, cell_state = self.lstmcell(context_and_embed, (hidden_state, cell_state))
        output = torch.log_softmax(self.fc_out(hidden_state), dim=1)
        return output, hidden_state, cell_state       
        
    def init_hidden(self):
        hidden_state = torch.zeros((1, self.hid_dim), device=device)
        cell_state = torch.zeros((1, self.hid_dim), device=device)
        
        torch.nn.init.xavier_normal_(hidden_state)
        torch.nn.init.xavier_normal_(cell_state)
        
        return hidden_state, cell_state
    

#Creating encoderdecoder_lstm    
class lstm_seq2seq(nn.Module):
    def __init__(self, source_input_dim, target_output_dim, source_emb_dim, target_emb_dim, hid_dim):
        super(lstm_seq2seq, self).__init__()
        self.source_input_dim = source_input_dim
        self.target_output_dim = target_output_dim
        self.encoder = Encoderlstm(source_input_dim, source_emb_dim, hid_dim).to(device)
        self.decoder = Decoderlstm(target_output_dim, target_emb_dim, hid_dim).to(device)
        
    def forward(self, source, target):
        encoder_hidden_state, encoder_cell_state = self.encoder.init_hidden()
        decoder_hidden_state, decoder_cell_state = self.decoder.init_hidden()
        
        outputs = torch.zeros((len(target), self.target_output_dim))
        
        for tensor in source:
            encoder_hidden_state, encoder_cell_state = self.encoder.forward(tensor, encoder_hidden_state, encoder_cell_state)
            
        for t in range(len(target)):
            output, decoder_hidden_state, decoder_cell_state = self.decoder.forward(target[t], encoder_hidden_state, decoder_hidden_state,
                                                                                    decoder_cell_state)
            outputs[t] = output
            
        return outputs

     