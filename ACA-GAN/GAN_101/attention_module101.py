import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        
        # Define convolution layers for query, key, and value
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        
        # Softmax for the attention mechanism
        self.softmax = nn.Softmax(dim=-2)
        
    def forward(self, x):
        batch_size, C, width, height = x.size()
        
        # Creating the Query, Key, and Value using the convolution layers
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        value = self.value_conv(x).view(batch_size, -1, width * height)
        
        # Computing the attention weights
        attention = self.softmax(torch.bmm(query, key))
        
        # Output after applying the attention mechanism
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(batch_size, C, width, height)
        
        return out