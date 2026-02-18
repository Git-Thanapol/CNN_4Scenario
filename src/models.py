import torch
import torch.nn as nn
import os
from .config import DROPOUT_RATE, DEVICE
try:
    from .ast_models import ASTModel
except ImportError:
    ASTModel = None # Handle missing timm gracefully during import

class SimpleCNN(nn.Module):
    """
    A lightweight CNN for spectrogram classification.
    Returns features in forward() for t-SNE visualization.
    """
    def __init__(self, n_classes: int, dropout_rate: float = DROPOUT_RATE):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_pool(x)
        features = x.view(x.size(0), -1) # Flatten: [Batch, 64]
        out = self.fc(self.dropout(features))
        return out, features

class CNN_MLP(SimpleCNN):
    """
    CNN backbone with MLP head instead of single linear layer.
    """
    def __init__(self, n_classes: int, dropout_rate: float = DROPOUT_RATE):
        super(CNN_MLP, self).__init__(n_classes, dropout_rate)
        # Replacing self.fc with MLP
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, n_classes)
        )

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class CNN_Attention(SimpleCNN):
    """
    CNN backbone with SE Attention Block.
    """
    def __init__(self, n_classes: int, dropout_rate: float = DROPOUT_RATE):
        super(CNN_Attention, self).__init__(n_classes, dropout_rate)
        # Re-define conv layers to insert SEBlock at the end or intermediate
        # Here we insert after last conv block before pooling
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Insert SE Block here
            SEBlock(64),
            nn.MaxPool2d(2)
        )

class VGG(nn.Module):
    """
    VGG-style CNN for 1-channel spectrograms.
    """
    def __init__(self, n_classes: int, dropout_rate: float = DROPOUT_RATE):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        features = x # VGG features are high dim, maybe bottleneck needed for t-SNE? Use pre-classifier
        # For consistency with training loop, return penultimate layer or flattened
        out = self.classifier(x)
        return out, features

class AFSC(nn.Module):
    """
    Adaptive Frequency Spectral Coefficient Module.
    Adapts 1D sequence from AST to 2D feature map for ResCNN.
    """
    def __init__(self, input_dim=768, out_channels=64, grid_h=12, grid_w=102):
        super(AFSC, self).__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        # 1x1 Conv to reduce dimentionality and "select coefficients"
        self.adapt_conv = nn.Conv2d(input_dim, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (B, N_patches, D)
        B, N, D = x.shape
        # Reshape to (B, D, H, W)
        # We assume N matches H*W approximately or we interpolate
        # AST base384 with 1024x128 input and stride 10 -> approx 102x12 patches = 1224.
        
        # Proper reshape requires knowing the exact grid.
        # If N != H*W, we might need to transpose and interpolate.
        # Let's trust the grid_h/w passed in or calculate dynamically?
        # For now, simplistic reshape:
        H, W = self.grid_h, self.grid_w
        
        # If mismatch, use interpolation
        if N != H * W:
            # Treat as 1D sequence and map to 2D?
            pass

        # Transpose to (B, D, N)
        x = x.transpose(1, 2) 
        # View as (B, D, sqrt(N), sqrt(N))? No, it's rect.
        # Fold it? 
        # AST patch embed uses row-major flattening.
        # So we can view as (B, D, H, W) directly if dimensions match.
        
        # HACK: Force reshape to roughly HxW, or interpolate 
        # Since we just need to feed ResCNN, spatial structure is preserved roughly.
        target_len = H * W
        if N > target_len:
             x = x[:, :, :target_len]
        elif N < target_len:
             x = torch.nn.functional.pad(x, (0, target_len - N))
             
        x = x.view(B, D, H, W)
        
        x = self.adapt_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ResCNN(nn.Module):
    """
    ResNet-style backend.
    """
    def __init__(self, in_channels=64, n_classes=4):
        super(ResCNN, self).__init__()
        # Simple ResNet-18 block structure
        self.layer1 = self._make_layer(in_channels, 64)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, n_classes)
        
    def _make_layer(self, in_c, out_c, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        features = x.view(x.size(0), -1)
        out = self.fc(features)
        return out, features

class AST_AFSC_ResCNN(nn.Module):
    def __init__(self, n_classes, ast_weights_path=None):
        super(AST_AFSC_ResCNN, self).__init__()
        if ASTModel is None:
            raise ImportError("timm library not installed. Cannot use AST model.")
            
        # Initialize AST
        # Hardcoded params based on user providing 'audioset_10_10_0.4593'
        self.ast = ASTModel(label_dim=527, fstride=10, tstride=10, 
                            input_fdim=128, input_tdim=1024, 
                            imagenet_pretrain=False, model_size='base384')
        
        if ast_weights_path and os.path.exists(ast_weights_path):
             print(f"Loading AST weights from {ast_weights_path}")
             checkpoint = torch.load(ast_weights_path, map_location=DEVICE)
             # Handle DataParallel if present in checkpoint keys
             if 'module.' in list(checkpoint.keys())[0]:
                 new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
                 self.ast.load_state_dict(new_state_dict, strict=False)
             else:
                 self.ast.load_state_dict(checkpoint, strict=False)
        
        # Freeze AST? User said "AST for Feature extraction only".
        for param in self.ast.parameters():
            param.requires_grad = False
            
        # Dimensions
        # AST base384 embed_dim = 768
        # patches = (1024-16)/10+1 * (128-16)/10+1 = 101 * 12 = 1212 patches
        self.afsc = AFSC(input_dim=768, out_channels=64, grid_h=12, grid_w=101)
        self.rescnn = ResCNN(in_channels=64, n_classes=n_classes)

    def forward(self, x):
        # x is spectrogram (B, 1, Hz, Time) or (B, Hz, Time)?
        # SimleCNN expects (B, 1, F, T) but usually dataset gives (B, 1, F, T)
        # AST expects (B, T, F)
        
        # Assumed input: (B, 1, 128, 1024) - typical logmel
        # Permute for AST
        x_ast = x.squeeze(1).transpose(1, 2) # (B, 1024, 128)
        
        ast_features = self.ast(x_ast) # (B, N, 768)
        
        afsc_out = self.afsc(ast_features) # (B, 64, 12, 101)
        
        out, features = self.rescnn(afsc_out)
        return out, features
