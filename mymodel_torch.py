import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def Qcell2tetra(indata):
    B, C, H, W = indata.size()
    indata = indata.reshape(B, C, H // 2, 2, W // 2, 2)
    indata = torch.mean(indata, dim=(3, 5))

    return indata.float()


# Stack Q-cell phase into 4 channels, likely Tetra pattern will reshape into bayer pattern
class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(N, C * self.bs ** 2, H // self.bs, W // self.bs)
        return x


# Unstack "SpaceToDepth"
class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.bs, self.bs, C // self.bs ** 2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        x = x.view(N, C // self.bs ** 2, H * self.bs, W * self.bs)
        return x


# Basic convolutional layer
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=False, prelu=False,
                 transpose=False, shuffle=False, groups=1):
        super(BasicConv, self).__init__()
        self.last_crop = False
        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = (kernel_size) // 2 - 1
            self.last_crop = kernel_size % 2 == 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias,
                                   groups=groups))
        else:
            if stride > 1:
                layers.append(torch.nn.ZeroPad2d((0, kernel_size // 2, 0, kernel_size // 2)))
                padding = 0
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias,
                                    groups=groups))
        if shuffle:
            layers.append(nn.ChannelShuffle(4))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        if prelu:
            layers.append(nn.PReLU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        z = self.main(x)
        if self.last_crop:
            z = z[:, :, :-1, :-1]
        return z


# Learnable filter layer
class DiffPadConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, n_phase, prelu=False, relu=False, norm=False,
                 groups=1, bias=False, drop=False):
        super(DiffPadConv, self).__init__()
        self.depth_to_sp = DepthToSpace(2)
        self.n_phase = n_phase
        self.n_layer = 2
        self.out_channel = out_channel
        padding_val = list(range(kernel_size // 2, kernel_size // 2 - n_phase, -1))

        layers = list()
        for i in range(self.n_phase):
            for j in range(self.n_phase):
                padding = (
                    padding_val[j], kernel_size - n_phase - padding_val[j], padding_val[i],
                    kernel_size - n_phase - padding_val[i])
                layers.append(torch.nn.ReplicationPad2d(padding))
                layers.append(
                    nn.Conv2d(in_channel, out_channel, kernel_size, padding=0, stride=n_phase, bias=bias,
                              groups=groups))
                if drop:
                    layers.append(nn.Dropout2d(p=0.02))
                if norm:
                    layers.append(nn.BatchNorm2d(out_channel))
                if relu:
                    layers.append(nn.ReLU(inplace=True))
                if prelu:
                    layers.append(nn.PReLU())
        if norm:
            self.n_layer += 1
        if relu:
            self.n_layer += 1
        if prelu:
            self.n_layer += 1
        if drop:
            self.n_layer += 1

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        B, C, H, W = x.size()

        z = torch.empty(B, 0, H // self.n_phase, W // self.n_phase, device=x.device)

        for i in range(self.n_phase * self.n_phase):
            y = x
            for j in range(self.n_layer):
                y = self.main[self.n_layer * i + j](y)

            z = torch.cat((z, y), dim=1)

        z = z.reshape(B, self.out_channel, self.n_phase, self.n_phase, H // self.n_phase, W // self.n_phase)
        z = z.permute(0, 1, 4, 2, 5, 3).contiguous()
        z = z.reshape(B, self.out_channel, H, W)

        return z


class SeparableMedianFilter(nn.Module):
    # 17 x 17 Median filter를 경량화하여 구현한 구조
    # Separable 컨셉으로 x축 먼저 median을 구한 후, y축으로 median을 구하는 형태
    # CPU에선 17칸의 큰 filter 사이즈에 의해 median 구하는게 느려 2칸씩 stride를 주도록 구현
    # HW에서 17칸 median filter에 대한 부담 여부에 따라 비슷하게 구현하면 될 것으로 보입니다.
    def __init__(self, kernel_size=(3, 3), stride=1, dilation=1):
        super(SeparableMedianFilter, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding_x = (kernel_size[0] // 2, kernel_size[0] // 2, 0, 0)
        self.padding_y = (0, 0, kernel_size[1] // 2, kernel_size[1] // 2)

    def forward(self, x):
        z = F.pad(x, self.padding_x, mode='replicate').squeeze(1)
        z = z.unfold(2, self.kernel_size[0], 1)[:, ::, self.stride // 2::self.stride, ::self.dilation]
        z = z.median(dim=-1)[0].unsqueeze(1)

        z = F.pad(z, self.padding_y, mode='replicate').squeeze(1)
        z = z.unfold(1, self.kernel_size[1], 1)[:, self.stride // 2::self.stride, ::, ::self.dilation]
        z = z.median(dim=-1)[0].unsqueeze(1)

        return z


def F_PeriodConv(input, weight, bias=None, period=4, stride=1, padding=0, groups=1):
    stride = stride
    kernel_size = weight.shape[-2:]
    b_in, c_in, h_in, w_in = input.shape
    batch_size = input.shape[0]
    h_out = (h_in + 2 * padding - (kernel_size[0] - 1) - 1) / stride + 1
    w_out = (w_in + 2 * padding - (kernel_size[1] - 1) - 1) / stride + 1
    h_out, w_out = int(h_out), int(w_out)
    inp_unf = F.unfold(input, (kernel_size[0], kernel_size[1]), padding=padding, stride=stride)
    inp_unf = torch.reshape(inp_unf, [inp_unf.shape[0], groups, inp_unf.shape[1] // groups, h_out // period, period,
                                      w_out // period, period])
    inp_unf = inp_unf.permute(0, 1, 4, 6, 2, 3, 5)
    inp_unf = inp_unf.reshape(inp_unf.size(0), inp_unf.size(1), inp_unf.size(2), inp_unf.size(3), inp_unf.size(4),
                              inp_unf.size(5) * inp_unf.size(6))
    kernel = weight.reshape(weight.size(0), weight.size(1), weight.size(2), weight.size(3),
                            weight.size(4) * weight.size(5) * weight.size(6)).transpose(3, 4)
    out_unf = inp_unf.transpose(4, 5).matmul(kernel).transpose(4, 5)
    out_unf = out_unf.reshape(out_unf.size(0), out_unf.size(1), out_unf.size(2), out_unf.size(3), out_unf.size(4),
                              h_out // period, w_out // period)
    if bias is not None:
        out_unf += bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    out_unf = out_unf.permute(0, 1, 4, 5, 2, 6, 3)
    out_ = torch.reshape(out_unf, (
        out_unf.shape[0], out_unf.shape[1] * out_unf.shape[2], out_unf.shape[3] * out_unf.shape[4],
        out_unf.shape[5] * out_unf.shape[6]))
    return out_


class ResBlock(nn.Module):
    def __init__(self, n_feats, n_feats1, kernel_size, bias=True, bn=False, act=nn.ReLU(True), relu=True, stride=1,
                 res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size // 2), bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res




class DeblurNet(nn.Module):
    def __init__(self):
        super(DeblurNet, self).__init__()
        self.feat_extract_tetra2 = nn.ModuleList([
            BasicConv(16, 32, kernel_size=3, prelu=True, stride=1),
            BasicConv(32, 32, kernel_size=3, prelu=True, stride=1),
            BasicConv(32, 32, kernel_size=3, prelu=True, stride=1),
            BasicConv(32, 32, kernel_size=3, prelu=True, stride=2),
            BasicConv(32, 32, kernel_size=3, prelu=True, stride=1),
            BasicConv(32, 32, kernel_size=3, prelu=True, stride=2),
            BasicConv(32, 32, kernel_size=3, prelu=True, stride=1),
            BasicConv(32, 32, kernel_size=3, prelu=True, stride=1),
            BasicConv(32, 32, kernel_size=3, prelu=True, stride=2, transpose=True),
            BasicConv(32, 32, kernel_size=3, prelu=True, stride=1),
            BasicConv(32, 32, kernel_size=3, prelu=True, stride=2, transpose=True),
            BasicConv(32, 32, kernel_size=3, prelu=True, stride=1),
            BasicConv(32, 32, kernel_size=3, prelu=True, stride=1),
            BasicConv(32, 32, kernel_size=3, prelu=True, stride=1),
            BasicConv(32, 32, kernel_size=3, prelu=True, stride=1),
            BasicConv(32, 48, kernel_size=3, prelu=False, stride=2, transpose=True),
        ])

        self.feat_extract_tetra2 = nn.ModuleList(self.feat_extract_tetra2)
        self.sp_to_depth = SpaceToDepth(2)
        self.depth_to_sp = DepthToSpace(2)

        self.feat_extract_tetra2.apply(self.weight_init)

    def forward(self, x):
        ##########################################################################
        # 2 x 2 binning
        x_in = Qcell2tetra(x)

        x_in = self.sp_to_depth(x_in)
        x_in = self.sp_to_depth(x_in)

        z = self.feat_extract_tetra2[0](x_in)
        z = self.feat_extract_tetra2[1](z) + z
        res1 = self.feat_extract_tetra2[2](z) + z
        z = self.feat_extract_tetra2[3](res1)
        res2 = self.feat_extract_tetra2[4](z) + z
        z = self.feat_extract_tetra2[5](res2)
        z = self.feat_extract_tetra2[6](z) + z
        z = self.feat_extract_tetra2[7](z) + z
        z = self.feat_extract_tetra2[8](z) + res2
        z = self.feat_extract_tetra2[9](z) + z
        z = self.feat_extract_tetra2[10](z) + res1
        z = self.feat_extract_tetra2[11](z) + z
        z = self.feat_extract_tetra2[12](z) + z
        z = self.feat_extract_tetra2[13](z) + z
        z = self.feat_extract_tetra2[14](z) + z

        z = self.feat_extract_tetra2[15](z)
        z = self.depth_to_sp(z)
        z = self.depth_to_sp(z)
        output = torch.clamp(z, -1, 1)

        return output

    def weight_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0.0, 0.5 * math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0, 0.01)
            m.bias.data = torch.ones(m.bias.data.size())


def build_net():
    return DeblurNet()
