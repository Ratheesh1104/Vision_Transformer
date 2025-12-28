import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, ima_size = 224, patch_size= 16 , in_channels = 3 , embed_dim = 769):
        super().__init__()
        self.num_patches = (ima_size // patch_size) ** 2

        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1,2)
        return x