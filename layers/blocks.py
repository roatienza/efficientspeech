
from einops import reduce, repeat
from torch import nn
import torch
import torch.nn.functional as F


class MixFFN(nn.Module):
    def __init__(
        self,
        dim,
        expansion_factor,
    ):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.mlp1 = nn.Linear(dim, hidden_dim)
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
        self.mlp2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()
        

    def forward(self, x):
        x = self.mlp1(x)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = self.act(x)
        x = self.mlp2(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3 * num_heads, bias=qkv_bias)
        self.proj = nn.Linear(dim * num_heads, dim)

    def forward(self, x, mask=None, pool=1):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C).permute(2, 0, 3, 1, 4)
        # qkv dim is [3, B, num_heads, N, C]
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_mask = None
        if mask is not None:
            if pool > 1:
                mod = mask.shape[-1] % pool
                if mod > 0:
                    pad = [0, int(pool-mod)]
                    mask = F.pad(mask, pad, value=True)
                mask = reduce(mask, 'b (n p) -> b n', 'max', p=pool)

            attn_mask = mask.unsqueeze(1).expand(-1, attn.shape[-1], -1)
            attn_mask = attn_mask.repeat(self.num_heads, 1, 1) 
            attn_mask = attn_mask.reshape(-1, self.num_heads, attn_mask.shape[-2], attn_mask.shape[-1])

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        
        if mask is not None:
            attn_mask = repeat(mask, 'b n -> b n a', a=x.shape[-1])

        return x, attn_mask


if __name__ == "__main__":
    x = torch.rand((32, 256, 100))
    y = SelfAttention(256)(x)
    print(y.size())
