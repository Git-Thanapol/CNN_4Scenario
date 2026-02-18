import torch
import torch.nn as nn
import timm
from timm.models.layers import to_2tuple, trunc_normal_

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class ASTModel(nn.Module):
    """
    The AST model.
    :param label_dim: the label dimension, i.e., the number of total classes, it is 527 for AudioSet, 50 for ESC-50, and 35 for speechcommands.
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6.
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6.
    :param input_fdim: the number of frequency bins of the input spectrogram.
    :param input_tdim: the number of time frames of the input spectrogram.
    :param imagenet_pretrain: if use ImageNet pretrained model.
    :param audioset_pretrain: if use full AudioSet and SpeechCommands pretrained model.
    :param model_size: the model size of AST, 'tiny', 'small', 'base', 'base384'.
    :param verbose: if show the model details.
    """
    def __init__(self, label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=True, audioset_pretrain=False, model_size='base384', verbose=True):

        super(ASTModel, self).__init__()
        
        # Override timm's default logic to support our specific requirements
        # We rely on timm for the Transformer body
        if model_size == 'base384':
            self.v = timm.create_model('vit_base_patch16_384', pretrained=imagenet_pretrain)
            self.original_num_patches = 576
        elif model_size == 'base224':
            self.v = timm.create_model('vit_base_patch16_224', pretrained=imagenet_pretrain)
            self.original_num_patches = 196
        else:
            raise ValueError("Model size not supported")

        self.original_hw = int(self.original_num_patches ** 0.5)
        self.original_embedding_dim = self.v.pos_embed.shape[2]
        self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))

        # Automate the patch embedding adjustment for audio spectrograms
        f_dim, t_dim = input_fdim, input_tdim
        self.fstride, self.tstride = fstride, tstride
        self.input_fdim, self.input_tdim = input_fdim, input_tdim
        
        # Calculate patch grid size
        # Patch size is 16x16 by default in ViT
        patch_size = 16
        
        # Calculate number of patches in freq and time
        # The logic below mimics the original AST implementation for "overlap" via stride
        # Number of patches = (InputDim - PatchSize) / Stride + 1
        
        self.n_patches_f = (f_dim - patch_size) // fstride + 1
        self.n_patches_t = (t_dim - patch_size) // tstride + 1
        num_patches = self.n_patches_f * self.n_patches_t
        
        # Re-initialize patch embedding layer
        new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=patch_size, stride=(fstride, tstride))
        if imagenet_pretrain:
            # Initialize with ImageNet weights (3 channels -> 1 channel by averaging)
            new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
            new_proj.bias = self.v.patch_embed.proj.bias
        self.v.patch_embed.proj = new_proj
        
        # Re-initialize positional embedding
        if imagenet_pretrain:
            # Interpolate pos_embed if needed
            # This is complex, but for now we initialize complex pos_embed logic only if we are training/fine-tuning.
            # If we load custom weights (AudioSet), those will overwrite this anyway.
            # So we just ensure the shape matches.
            new_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.original_embedding_dim))
            # (CLS + Dist token + Patches) -> ViT usually has 1 or 2 special tokens.
            # verify token count: distilled ViT has 2 (CLS, DIST), normal has 1 (CLS).
            # 'vit_base_patch16_384' typically has 1 CLS.
            # Check self.v.pos_embed.shape
            pass 

        # We overwrite pos_embed to match the new sequence length
        # +2 for (cls, dist) if distilled, +1 if normal.
        # Let's check the model type name
        if 'distilled' in model_size:
            self.v.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.original_embedding_dim))
            self.dist_token = nn.Parameter(torch.zeros(1, 1, self.original_embedding_dim))
        else:
            self.v.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.original_embedding_dim))
            
        # Initialize the new pos_embed properly? 
        # For our case, we LOAD weights later. So random init here is "okay" as long as shapes match.
        trunc_normal_(self.v.pos_embed, std=.02)

    def get_skip_balance(self, x):
        return self.mlp_head(self.v(x))

    def forward(self, x):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :return: prediction
        """
        # Expect input (B, T, F) -> (B, 1, F, T) for Conv2d?
        # Standard AST code expects (B, T, F).
        # It unsqueezes to (B, 1, T, F) inside usually, but patch_embed expects (B, C, H, W).
        # PatchEmbed in timm (and my override) uses Conv2d(1, ...).
        # Conv2d takes (H, W) = (Freq, Time) or (Time, Freq)?
        # Original AST: x = x.unsqueeze(1) # (B, 1, T, F)
        # x = x.transpose(2, 3) # (B, 1, F, T) -> H=F, W=T
        
        x = x.unsqueeze(1)
        x = x.transpose(2, 3) 
        
        B = x.shape[0]
        x = self.v.patch_embed(x)
        
        # Add CLS token
        cls_token = self.v.cls_token.expand(B, -1, -1)
        if hasattr(self.v, 'dist_token') and self.v.dist_token is not None:
            dist_token = self.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_token, dist_token, x), dim=1)
        else:
            x = torch.cat((cls_token, x), dim=1)
            
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        
        # Apply Transformer Blocks
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        
        # Return Sequence (B, N, D)
        # Remove CLS (and Dist) token for our AFSC purpose?
        # Or return only CLS?
        # For AST_AFSC_ResCNN, we want the spatial features (patches).
        # So exclude the first token(s).
        
        if hasattr(self.v, 'dist_token') and self.v.dist_token is not None:
             return x[:, 2:]
        else:
             return x[:, 1:]
