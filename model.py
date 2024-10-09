import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x
    
class TransformerEncoderModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoder = PositionalEncoding(config.d_model, config.max_seq_length)
        encoder_layers = nn.TransformerEncoderLayer(config.d_model, config.nhead, config.dim_feedforward, config.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, config.num_encoder_layers)
        self.d_model = config.d_model
        self.avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.linear1 = nn.Linear(config.d_model, config.d_model // 3)
        self.linear2 = nn.Linear(config.d_model // 3, 2)

    def forward(self, inputs, src_mask=None, src_key_padding_mask=None):        
        output = self.embedding(inputs) * math.sqrt(self.d_model) 
        # shape [batch_size, max_batch_sequence_length, d_model]
        output = self.pos_encoder(output)

        # shape [max_batch_sequence_length, batch_size, d_model]
        output = output.permute(1, 0, 2)
        output = self.transformer_encoder(output, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        
        # reshape for average pooling
        # change from [seq_len, batch_size, d_model] to [batch_size, d_model, seq_len]
        output = output.permute(1, 2, 0)  
        output = self.avg_pooling(output)
        output = output.squeeze(-1)  # remove the last dimension after pooling
        
        # shape [batch_size, d_model]
        output = F.relu(self.linear1(output))
        # shape [batch_size, 2]
        output = self.linear2(output)
        return output
    
    

    
