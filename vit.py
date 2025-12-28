import torch.nn as nn
import torch
from .patching import PatchEmbedding
from .transformer import TransformerEncoderBlock

class VisionTransformer(nn.Module):
    def __init__(
            self,
            img_size = 224,
            patch_size = 16,
            num_classes = 10,
            embed_dim = 768,
            depth = 6,
            num_head = 8,
            mlp_dim = 1024
    ):
        
        super().__init__()

        self.patch_embed = PatchEmbedding(torch.zero(1,1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zero(1, self.patch_embed.num_patches +1, embed_dim)
        )

        self.encoder = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim, num_head, mlp_dim
            ) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)

        x =self.patch_embed(x)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = x + self.pos_embed
        x = x.transpose(0,1)

        for block in self.encoder:
            x = block(x)

        x = x.transpose(0,1)

        x = self.norm(x)

        cls_output = x[:, 0]
        return self.head(cls_output)