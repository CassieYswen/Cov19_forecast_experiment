import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class FlexibleRNNModel(nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size, output_size, num_layers=1, dropout=0.2, bidirectional=True):
        super(FlexibleRNNModel, self).__init__()
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define the type of RNN (LSTM, GRU, or Vanilla RNN)
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)
        else:  # Default to vanilla RNN
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True, nonlinearity='tanh', bidirectional=bidirectional)

        # Define the output layer
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        # Initialize hidden state
        x = self.dropout(x)
        if self.rnn_type == 'LSTM' or self.rnn_type == 'GRU':
            #h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            #c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.rnn(x) #, (h0, c0)
        else:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.rnn(x, h0)

        # Pass the output of the last time step to the classifier
        #activation function?
        out = self.fc(out[:, -1, :])
        return out

# Example usage
# model = FlexibleRNNModel(rnn_type='GRU', input_size=10, hidden_size=20, output_size=1)
# model = FlexibleRNNModel(rnn_type='LSTM', input_size=10, hidden_size=20, output_size=1)
# model = FlexibleRNNModel(rnn_type='RNN', input_size=10, hidden_size=20, output_size=1)

#define the loss function as weighted MSE
def custom_loss(y_pred, y_true):
    # Ensure y_pred is not zero
    y_pred = y_pred.clamp(min=1e-8)
    return torch.mean((y_pred - y_true)** 2)

class TimeSeriesDataset(Dataset):
    def __init__(self, dataframe, sequence_length):
        self.sequence_length = sequence_length
        self.data = dataframe
        # Shift and take logarithm of R, fill missing values with log(3)
        self.data['log_R_t-1'] = np.log(self.data['R'].shift(1)).fillna(np.log(3))
        self.data['log_R'] = np.log(self.data['R'])
        #import pdb; pdb.set_trace()
        self.features = self.data[['log_R_t-1', 'Z1', 'Z2']].values
        #self.features = self.data[['log_R_t-1' ]].values
        self.target = self.data['log_R'].values
        #min max each feature of self.features
        #import pdb; pdb.set_trace()
        self.features = (self.features - self.features.min(axis=0)) / (self.features.max(axis=0) - self.features.min(axis=0))
    def __len__(self):
        print ('length', len(self.data) - self.sequence_length)
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        print ('idx', idx)
        # start_idx = max(idx, 1)  # To handle the case when idx=0 after shifting
        # # features = self.data[['log_R_t-1', 'Z1', 'Z2']].iloc[start_idx-1:start_idx-1+self.sequence_length-1].values
        # # target = self.data['log_R'].iloc[start_idx-1+self.sequence_length]
        # features = self.features[start_idx-1:start_idx-1+self.sequence_length-1]
        # target = self.target[start_idx-1+self.sequence_length]
        start_idx = idx
        features = self.features[start_idx :start_idx +self.sequence_length-1]
        target = self.target[start_idx +self.sequence_length]
        #import pdb; pdb.set_trace()
        return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
