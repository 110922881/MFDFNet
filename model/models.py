import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True):
        super(DepthwiseSeparableConv2d, self).__init__()

        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=in_channels, bias=bias
        )

        # Pointwise convolution
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1,
            stride=1, padding=0, dilation=1, bias=bias
        )

    def forward(self, x):
        # Apply depthwise convolution
        x = self.depthwise(x)
        # Apply pointwise convolution
        x = self.pointwise(x)

        return x

class SP(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size//2)
    def forward(self, x):
        return self.pool(x)

class ILK(nn.Module):
    def __init__(self, c1, c2, c3, c4):  # c1=3, c2=64, c3=64, c4=32
        super().__init__()
        self.cv1 = nn.Conv2d(c1, c3, 1, 1)
        self.cv2 = DepthwiseSeparableConv2d(c1, c4, kernel_size=3, stride=1, padding=1)
        self.cv3 = nn.Conv2d(c3 + 3*c4, c2, 1, 1)
        self.cv4 = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.cv5 = SP(5)
        self.cv6 = nn.Conv2d(c3, c4, kernel_size=5, stride=1, padding=2)
        self.cv7 = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        y = self.cv1(x)                # (B,c3,H,W)
        y1 = self.cv4(y)               # (B,c4,H,W)
        y2 = self.cv7(y1)              # (B,c4,H,W)
        y3 = self.cv6(y)               # (B,c4,H,W)
        y5 = self.cv5(x)               # (B,c1,H,W)
        y51 = self.cv2(y5)             # (B,c4,H,W)
        out = torch.cat([y, y2, y3, y51], dim=1)  # (B, c3+3*c4, H,W)
        return self.cv3(out)           # (B,c2,H,W)

class FCPResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels * 4, kernel_size=1)
        self.conv3x3_x2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3x3_x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3x3_x4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.final_conv1x1 = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = None

    def forward(self, x):
        residual = x
        x = self.conv1x1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        y1 = x1
        y2 = F.relu(self.conv3x3_x2(x2))
        y3 = F.relu(self.conv3x3_x3(y2 + x3))
        y4 = F.relu(self.conv3x3_x4(y3 + x4))
        y = torch.cat([y1, y2, y3, y4], dim=1)
        y = self.final_conv1x1(y)
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
        return y + residual

class SPDConv(nn.Module):
    def __init__(self, in_channels, out_channels, downscale_factor=2):
        super().__init__()
        self.downscale_factor = downscale_factor
        self.conv = nn.Conv2d(in_channels * (downscale_factor ** 2), out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape
        f = self.downscale_factor
        x = x.reshape(B, C, H // f, f, W // f, f).permute(0, 3, 5, 1, 2, 4).reshape(B, C * f * f, H // f, W // f)
        return self.conv(x)

class LGF(nn.Module):
    def __init__(self, in_channels=3, ilk_out=64, mid_channels=64):
        super().__init__()

        self.ilk = ILK(c1=in_channels, c2=ilk_out, c3=64, c4=32)

        self.branch1_conv = nn.Conv2d(ilk_out, mid_channels, kernel_size=1)
        self.branch2_conv = nn.Conv2d(ilk_out, mid_channels, kernel_size=1)

        self.fcp_resblock = FCPResBlock(mid_channels, mid_channels)

        self.concat_conv = nn.Conv2d(mid_channels * 2, ilk_out, kernel_size=1, padding=1)

        self.residual_adapt = nn.Conv2d(in_channels, ilk_out, kernel_size=1)

        self.spd_conv = SPDConv(ilk_out, ilk_out, downscale_factor=2)

    def forward(self, x):
        identity = x

        ilk_out = self.ilk(x)                     # [B, ilk_out, H, W]

        branch1 = self.branch1_conv(ilk_out)      # [B, mid_channels, H, W]
        branch2 = self.branch2_conv(ilk_out)      # [B, mid_channels, H, W]
        branch2 = self.fcp_resblock(branch2)      # [B, mid_channels, H, W]

        concat = torch.cat([branch1, branch2], dim=1)  # [B, mid_channels*2, H, W]

        conv_out = self.concat_conv(concat)            # [B, ilk_out, H, W]

        residual_path = self.residual_adapt(identity)  # [B, ilk_out, H, W]

        out = conv_out + residual_path                 # [B, ilk_out, H, W]

        output = self.spd_conv(out)                    # [B, ilk_out, H/2, W/2]
        return output

#UID
class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class RFAConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size

        self.get_weight = nn.Sequential(nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),
                                        nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=1,
                                                  groups=in_channel, bias=False))
        self.generate_feature = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=kernel_size, padding=kernel_size // 2,
                      stride=stride, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU())

        self.conv = Conv(in_channel, out_channel, k=kernel_size, s=kernel_size, p=0)

    def forward(self, x):
        b, c = x.shape[0:2]
        weight = self.get_weight(x)
        h, w = weight.shape[2:]
        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2)  # b c*kernel**2,h,w ->  b c k**2 h w
        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h,
                                                w)  # b c*kernel**2,h,w ->  b c k**2 h w
        weighted_data = feature * weighted
        conv_data = rearrange(weighted_data, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              # b c k**2 h w ->  b c h*k w*k
                              n2=self.kernel_size)
        return self.conv(conv_data)

class RefConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, groups=1,
                 map_k=3):
        super(RefConv, self).__init__()
        assert map_k <= kernel_size
        self.origin_kernel_shape = (out_channels, in_channels // groups, kernel_size, kernel_size)
        self.register_buffer('weight', torch.zeros(*self.origin_kernel_shape))
        G = in_channels * out_channels // (groups ** 2)
        self.num_2d_kernels = out_channels * in_channels // groups
        self.kernel_size = kernel_size
        self.convmap = nn.Conv2d(in_channels=self.num_2d_kernels,
                                 out_channels=self.num_2d_kernels, kernel_size=map_k, stride=1, padding=map_k // 2,
                                 groups=G, bias=False)
        #nn.init.zeros_(self.convmap.weight)
        self.bias = None#nn.Parameter(torch.zeros(out_channels), requires_grad=True)     # must have a bias for identical initialization
        self.stride = stride
        self.groups = groups
        if padding is None:
            padding = kernel_size // 2
        self.padding = padding

    def forward(self, inputs):
        origin_weight = self.weight.view(1, self.num_2d_kernels, self.kernel_size, self.kernel_size)
        kernel = self.weight + self.convmap(origin_weight).view(*self.origin_kernel_shape)
        return F.conv2d(inputs, kernel, stride=self.stride, padding=self.padding, dilation=1, groups=self.groups, bias=self.bias)


class UID(nn.Module):
    """Enhanced DynamicConv with improved channel attention and feature refinement."""

    def __init__(self, c1, c2, k1=3, k2=1, s1=2, s2=1, d1=1, d2=1, act=True):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = RefConv(c1, self.c, k1, s1)
        self.cv2 = RFAConv(c1, self.c, k2, s2)

        self.se_cv1 = SEBlock(self.c, reduction=8)
        self.se_x2 = SEBlock(self.c // 2, reduction=7)
        self.se_x3 = SEBlock(self.c // 2, reduction=6)

        self.conv_x2 = Conv(self.c//2, self.c//2, 1, 1, act=act)
        self.conv_x3 = Conv(self.c//2, self.c//2, 1, 1, act=act)

        self.fusion_conv = Conv(c2, c2, 1, 1, act=act)

    def forward(self, x):
        x1 = self.se_cv1(self.cv1(x))

        x = self.cv2(x)
        x2, x3 = x.chunk(2, dim=1)

        x2 = F.max_pool2d(x2, 3, stride=2, padding=1)
        x2 = self.conv_x2(x2)
        x2 = self.se_x2(x2)

        x3 = F.avg_pool2d(x3, 3, stride=2, padding=1)
        x3 = self.conv_x3(x3)
        x3 = self.se_x3(x3)

        return self.fusion_conv(torch.cat([x1, x2, x3], dim=1))

#LCN
class ECA(nn.Module):

    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        self.channels = channels
        kernel_size = self.adaptive_kernel_size(channels, gamma, b)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def adaptive_kernel_size(self, c, gamma, b):
        t = int(abs((math.log2(c) + b) / gamma))
        return t if t % 2 else t + 1

    def forward(self, x):
        B, C, H, W = x.size()

        y = x.mean((2, 3), keepdim=True)
        y = y.view(B, 1, C)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.view(B, C, 1, 1)

        return x * y.expand_as(x)

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # B C H W → B H W C
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # B H W C → B C H W
        return x + input

class LCN(nn.Module):
    def __init__(self, dim, depth=10):
        super().__init__()
        self.blocks = nn.Sequential(*[ConvNeXtBlock(dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.act = nn.GELU()

        self.se = ECA(dim)

    def forward(self, x):
        out = self.blocks(x)
        out = out + x
        out = out.permute(0, 2, 3, 1).contiguous()  # B C H W → B H W C
        out = self.norm(out)
        out = self.act(out)
        out = out.permute(0, 3, 1, 2).contiguous()  # B H W C → B C H W

        se_weight = self.se(out)
        out = out * se_weight.expand_as(out)

        return out
