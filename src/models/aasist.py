import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # FIX: Renamed 'F' to 'feat_dim' to avoid overwriting the F import
        B, N, feat_dim = h.size()
        
        # 1. Linear Transform
        h_prime = torch.matmul(h, self.W).view(B, N, self.n_heads, self.out_features)

        # 2. Attention Mechanism
        # Split 'a' into source and target weights
        a_src = self.a[:self.out_features, :] 
        a_src = a_src.permute(1, 0) # (Heads, Features)
        
        # Calculate scores
        scores = (h_prime * a_src.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        e = self.leakyrelu(scores)
        
        # Softmax & Dropout
        attention = F.softmax(e, dim=1).unsqueeze(-1) 
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        
        # 3. Weighted Sum
        out = h_prime * attention
        return out.sum(dim=2)

class AASIST(nn.Module):
    def __init__(self):
        super(AASIST, self).__init__()
        
        self.sinc_conv = nn.Conv1d(1, 70, kernel_size=128, stride=1, padding=64)
        
        self.encoder = nn.Sequential(
            nn.MaxPool1d(3),
            nn.BatchNorm1d(70), nn.LeakyReLU(0.3),
            nn.Conv1d(70, 32, 3, padding=1), 
            nn.BatchNorm1d(32), nn.LeakyReLU(0.3),
            nn.Conv1d(32, 64, 3, padding=1), 
            nn.MaxPool1d(3)
        )
        
        self.gat = GraphAttentionLayer(in_features=64, out_features=32, n_heads=4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        x = self.sinc_conv(x)
        x = self.encoder(x)
        x = x.transpose(1, 2) # Time as Nodes
        x = self.gat(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        return self.fc(x), None