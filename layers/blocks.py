
from einops import rearrange, reduce, repeat
from torch import nn, einsum
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
        x = rearrange(x, 'b n c -> b c n')
        x = self.conv(x)
        x = rearrange(x, 'b c n-> b n c')
        x = self.act(x)
        x = self.mlp2(x)
        return x



class EfficientSelfAttention(nn.Module):
    def __init__(self,
                 dim,
                 head=2):
        super().__init__()

        self.head = head
        self.scale = (dim // head) ** -0.5

        self.to_q = nn.Linear(dim, head * dim)
        self.to_k = nn.Linear(dim, head * dim)
        self.to_v = nn.Linear(dim, head * dim)
        self.merge = nn.Linear(head * dim, dim)

    def forward(self, x, mask=None, pool=1):
        attn_mask = None
        head = self.head

        q, k, v = (self.to_q(x), self.to_k(x), self.to_v(x))
        q, k, v = map(lambda t: rearrange(t, 'b n (h c) -> (b h) n c', h=head), (q, k, v))

        attn = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if mask is not None:
            if pool > 1:
                mod = mask.shape[-1] % pool
                if mod > 0:
                    mask = F.pad(mask, (0, pool-mod), value=True)
                mask = reduce(mask, 'b (n p) -> b n', 'max', p=pool)
            attn_mask = mask.unsqueeze(1).expand(-1, attn.shape[-1], -1)
            attn_mask = attn_mask.repeat(head, 1, 1) 
            #attn = attn.masked_fill(attn_mask, torch.iinfo(int).min)
            attn = attn.masked_fill(attn_mask, -2**15)

        attn = attn.softmax(dim = -1)

        attn = einsum('b i j, b j d -> b i d', attn, v)
        attn = rearrange(attn, '(b h) n c -> b n (h c)', h=head, n=x.shape[-2])
        attn = self.merge(attn)

        if mask is not None:
            attn_mask = repeat(mask, 'b n -> b n a', a=attn.shape[-1])
        return attn, attn_mask

if __name__ == "__main__":
    x = torch.rand((32, 256, 100))
    y = EfficientSelfAttention(256, reduction=2, head=2)(x)
    print(y.size())
