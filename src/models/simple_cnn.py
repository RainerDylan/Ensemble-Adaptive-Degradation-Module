"""
Simple CNN baseline model for audio deepfake detection
This is a lightweight model to verify the pipeline works before implementing full LCNN/Res2Net/AASIST
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Lightweight CNN for binary classification (bonafide vs spoof)
    
    Architecture:
        - 3 convolutional blocks with batch norm and max pooling
        - Global average pooling
        - Fully connected classifier
    
    Input: (batch_size, 1, 500, 80) - (batch, channels, time, frequency)
    Output: (batch_size, 2) - logits for [bonafide, spoof]
    """
    
    def __init__(self, dropout_rate=0.3):
        super(SimpleCNN, self).__init__()
        
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=2
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Block 2
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Block 3
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Fully connected classifier
        self.fc = nn.Linear(128, 2)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 1, 500, 80)
            
        Returns:
            logits: Output tensor of shape (batch_size, 2)
            embedding: Feature embedding before classifier (batch_size, 128)
        """
        # Convolutional Block 1
        x = self.conv1(x)           # (B, 32, 500, 80)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)           # (B, 32, 250, 40)
        
        # Convolutional Block 2
        x = self.conv2(x)           # (B, 64, 250, 40)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)           # (B, 64, 125, 20)
        
        # Convolutional Block 3
        x = self.conv3(x)           # (B, 128, 125, 20)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)           # (B, 128, 62, 10)
        
        # Global Average Pooling
        x = self.global_pool(x)     # (B, 128, 1, 1)
        embedding = x.view(x.size(0), -1)  # (B, 128)
        
        # Dropout and classification
        x = self.dropout(embedding)
        logits = self.fc(x)         # (B, 2)
        
        return logits, embedding
    
    def count_parameters(self):
        """Count total and trainable parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params


def test_model():
    """Test function to verify model architecture"""
    print("Testing SimpleCNN model...")
    
    # Create model
    model = SimpleCNN(dropout_rate=0.3)
    
    # Count parameters
    total, trainable = model.count_parameters()
    print(f"\nModel Parameters:")
    print(f"  Total: {total:,}")
    print(f"  Trainable: {trainable:,}")
    
    # Create dummy input (batch_size=4, channels=1, time=500, freq=80)
    dummy_input = torch.randn(4, 1, 500, 80)
    print(f"\nInput shape: {dummy_input.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits, embedding = model(dummy_input)
    
    print(f"\nOutput shapes:")
    print(f"  Logits: {logits.shape}")  # Should be (4, 2)
    print(f"  Embedding: {embedding.shape}")  # Should be (4, 128)
    
    # Test predictions
    probs = F.softmax(logits, dim=1)
    predictions = torch.argmax(probs, dim=1)
    
    print(f"\nSample predictions:")
    print(f"  Logits: {logits[0].tolist()}")
    print(f"  Probabilities: {probs[0].tolist()}")
    print(f"  Predicted class: {predictions[0].item()}")
    
    print("\n✓ Model test passed!")
    
    return model


if __name__ == "__main__":
    model = test_model()
