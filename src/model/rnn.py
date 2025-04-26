import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence

# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x, x_lengths):
        x_pack = pack_padded_sequence(x, x_lengths.to('cpu'), batch_first=True)
        out, _ = self.lstm(x_pack)

        # Decode the hidden state of the last time step
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = self.fc(out[:, -1, :])
        return out

class RNNAttentionMultiloss(nn.Module):
    def __init__(self, small_dim, hidden_size, num_layers, num_classes):
        super(RNNAttentionMultiloss, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.linear_affect = nn.Linear(8192, small_dim)
        self.lstm = nn.LSTM(small_dim + 49, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size + 200, num_classes)
        self.embedding_m = nn.Linear(3, 200)
        self.zero = torch.zeros(1, 200).cuda()
        self.attention = nn.Linear(3, 200)
    
    def forward(self, face, affect, meta, x_lengths):
        affect = self.linear_affect(affect)
        x = torch.cat([face, affect], dim=2)

        x_pack = pack_padded_sequence(x, x_lengths.to('cpu'), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(x_pack)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        attention = self.attention(meta)
        meta = self.embedding_m(meta)
        attention = torch.nn.functional.softmax(attention, dim=-1)
        out = out[:, -1, :] * attention

        out = self.fc(torch.cat([out, meta], dim=1))
        out_m = self.fc(torch.cat([self.zero, meta], dim=1))

        return out, out_m
