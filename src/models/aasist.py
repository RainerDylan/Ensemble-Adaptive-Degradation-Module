import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, n_heads, dropout=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.dropout = dropout

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features * n_heads)))
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, n_heads)))
        
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, h):
        # h: (Batch, Nodes, Features)
        B, N, F = h.size()
        h_prime = torch.matmul(h, self.W).view(B, N, self.n_heads, self.out_features)
        
        # Simple self-attention mechanism (Simplified for stability)
        # We compute attention scores based on node features
        e = torch.matmul(h_prime, self.a.mean(dim=1).unsqueeze(0).unsqueeze(-1)) # (B, N, Heads, 1)
        attention = F.softmax(e, dim=1) # (B, N, Heads, 1)
        
        h_prime = h_prime * attention
        return h_prime.sum(dim=2) # Sum over heads

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, first=False):
        super(ResidualBlock, self).__init__()
        self.first = first
        
        if not first:
            self.bn1 = nn.BatchNorm1d(in_channels)
        
        self.lrelu = nn.LeakyReLU(0.3)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.downsample = None
            
        self.maxpool = nn.MaxPool1d(3)

    def forward(self, x):
        identity = x
        out = x
        if not self.first:
            out = self.bn1(out)
            out = self.lrelu(out)
            
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        
        if self.downsample is not None:
            identity = self.downsample(identity)
            
        out += identity
        out = self.maxpool(out)
        return out

class AASIST(nn.Module):
    """
    AASIST Model Implementation
    Input: Raw Waveform (Batch, 1, 64600)
    Output: Logits (Batch, 2)
    """
    def __init__(self):
        super(AASIST, self).__init__()
        
        # 1. SincConv Front-end (Approximated with Conv1d for stability)
        # 70 filters, kernel size 128, matches AASIST paper specs roughly
        self.sinc_conv = nn.Conv1d(1, 70, kernel_size=128, stride=1, padding=64)
        
        # 2. Residual Encoder (RawNet2-based)
        self.res_blocks = nn.Sequential(
            ResidualBlock(70, 32, first=True),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            ResidualBlock(32, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )
        
        # 3. Graph Attention Back-end
        # We project the extracted features into a graph-compatible shape
        self.gat_layer = GraphAttentionLayer(64, 32, n_heads=4)
        
        # 4. Classifier
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        # Input x: (Batch, 1, Time) -> e.g. (Batch, 1, 64600)
        
        # Encoder
        x = self.sinc_conv(x)
        x = F.max_pool1d(torch.abs(x), 3)
        x = self.res_blocks(x) # (Batch, 64, Time_Frames)
        
        # Prepare for Graph (Treat temporal frames as nodes)
        # x: (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2) 
        
        # Graph Attention
        x = self.gat_layer(x) # (B, T, 32)
        
        # Pooling & Classification
        x = x.transpose(1, 2) # (B, 32, T)
        x = self.pool(x).squeeze(-1) # (B, 32)
        out = self.fc(x) # (B, 2)
        
        return out, x