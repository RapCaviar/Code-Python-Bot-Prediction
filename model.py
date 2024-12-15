import torch
import torch.nn as nn

# Combined LSTM and GRU model architecture
class LSTM_GRU(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, output_size):
        super(LSTM_GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True)
        
        
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        
        
        self.fc = nn.Linear(hidden_dim, output_size)
        
    def forward(self, x):
        h0_lstm = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0_lstm = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        h0_gru = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        
        
        out, (hn_lstm, cn_lstm) = self.lstm(x, (h0_lstm.detach(), c0_lstm.detach()))
        
        
        out, hn_gru = self.gru(out, h0_gru.detach())
        
    
        out = self.fc(out[:, -1, :])
        return out
