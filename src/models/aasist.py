import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, n_heads):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features * n_heads)))
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, n_heads)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h):
        B, N, F = h.size()
        h_prime = torch.matmul(h, self.W).view(B, N, self.n_heads, self.out_features)
        e = torch.matmul(h_prime, self.a.mean(dim=1).unsqueeze(0).unsqueeze(-1))
        return (h_prime * F.softmax(e, dim=1)).sum(dim=2)

class AASIST(nn.Module):
    def __init__(self):
        super(AASIST, self).__init__()
        self.sinc_conv = nn.Conv1d(1, 70, 128, stride=1, padding=64)
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(70), nn.LeakyReLU(0.3),
            nn.Conv1d(70, 32, 3, padding=1), nn.MaxPool1d(3),
            nn.BatchNorm1d(32), nn.LeakyReLU(0.3),
            nn.Conv1d(32, 64, 3, padding=1), nn.MaxPool1d(3)
        )
        self.gat = GraphAttentionLayer(64, 32, n_heads=4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        x = self.sinc_conv(x)
        x = self.encoder(x)
        x = x.transpose(1, 2) # (B, T, C)
        x = self.gat(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        return self.fc(x), None