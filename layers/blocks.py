
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
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = self.act(x)
        x = self.mlp2(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False):
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
        #q, k, v = map(lambda t: rearrange(t, 'b n (h c) -> (b h) n c', h=head), (q, k, v))
        q = rearrange(q, 'b n (h c) -> (b h) n c', h=head)
        k = rearrange(k, 'b n (h c) -> (b h) n c', h=head)
        v = rearrange(v, 'b n (h c) -> (b h) n c', h=head)

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

        #print("E Mask:", mask.shape)
        #print("E Attn 1:", attn_mask.shape)
        if mask is not None:
            attn_mask = repeat(mask, 'b n -> b n a', a=attn.shape[-1])
        #print("E Attn 2:", attn_mask.shape)
        #print("E x:", x.shape)
        #exit(0)
        return attn, attn_mask

class Attention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None, pool=1):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # qkv dim is [3, B, num_heads, N, C//num_heads]
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        #print("Attn:", attn.shape) 
        attn_mask = None
        if mask is not None:
            if pool > 1:
                mod = mask.shape[-1] % pool
                if mod > 0:
                    pad = [0, int(pool-mod)]
                    #mask = F.pad(mask, pad, value=1.0)
                    mask = F.pad(mask, pad, value=True)
                #print("Pool:", pool)
                #print("Mask shape:", mask.shape)

                mask = reduce(mask, 'b (n p) -> b n', 'max', p=pool)

                #print("Reduced mask shape:", mask.shape)
                #exit(0)
            attn_mask = mask.unsqueeze(1).expand(-1, attn.shape[-1], -1)
            #print("Shape:", attn_mask.shape)
            attn_mask = attn_mask.repeat(self.num_heads, 1, 1) 
            attn_mask = attn_mask.reshape(-1, self.num_heads, attn_mask.shape[-2], attn_mask.shape[-1])
            #print("Shape heads:", attn_mask.shape)
           
            #attn = attn.masked_fill(attn_mask, -2**15)

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        #print("Mask:", mask.shape)
        #print("Attn 1:", attn_mask.shape)
        if mask is not None:
            attn_mask = repeat(mask, 'b n -> b n a', a=x.shape[-1])

        #print("Attn 2:", attn_mask.shape)
        #print("x:", x.shape)
        #exit(0)
        return x, attn_mask

class EfficientSelfAttention_(nn.Module):
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
        #q, k, v = map(lambda t: rearrange(t, 'b n (h c) -> (b h) n c', h=head), (q, k, v))
        q = rearrange(q, 'b n (h c) -> (b h) n c', h=head)
        k = rearrange(k, 'b n (h c) -> (b h) n c', h=head)
        v = rearrange(v, 'b n (h c) -> (b h) n c', h=head)

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
