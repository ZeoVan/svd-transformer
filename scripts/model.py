import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, EMBED_SIZE, EMBED_DIM, TRANS_HEAD, TRANS_FWD, TRANS_DROP, TRANS_ACTIV, TRANS_LAYER, LSTM_SIZE, LSTM_LAYER, LSTM_BIDIR, FC_DROP, FC_OUT):
        super(Transformer,self).__init__()
        
        self.embed = nn.Embedding(num_embeddings=EMBED_SIZE,
                                  embedding_dim=EMBED_DIM)
        self.encode_layer = nn.TransformerEncoderLayer(d_model=EMBED_DIM,
                                                       nhead=TRANS_HEAD,
                                                       dim_feedforward=TRANS_FWD,
                                                       dropout=TRANS_DROP,
                                                       activation=TRANS_ACTIV)
        self.trans_encoder = nn.TransformerEncoder(self.encode_layer,
                                                   num_layers=TRANS_LAYER)
        self.lstm1 = nn.LSTM(input_size=EMBED_DIM,
                            hidden_size=LSTM_SIZE,
                            num_layers=LSTM_LAYER,
                            batch_first=True,
                            bidirectional=LSTM_BIDIR)
        self.dropout = nn.Dropout(FC_DROP)

        if LSTM_BIDIR is True:
          self.fc1 = nn.Linear(LSTM_SIZE*2,FC_OUT)
        else:
          self.fc1 = nn.Linear(LSTM_SIZE,FC_OUT)
    
    def forward(self, x):
        x = self.embed(x)
        x = self.trans_encoder(x)
        output, (hidden, cell) = self.lstm1(x)
        x = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        x = self.fc1(x)
        return(x)