import torch.nn as nn

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim , num_heads, mlp_dim, droupot = 0.1):
        super.__init__(self)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=droupot
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(droupot),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(droupot)
        )

    def forward(self, x):

        attn_out, _ = self.attn(
            self.norm1(x),
            self.norm1(x),
            self.norm(x)
        )

        x = x + attn_out
        x = x + self.mlp(self.norm2(x))

        return x