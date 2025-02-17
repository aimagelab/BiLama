import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import TypedDict
import math

from modules.base import get_activation
from modules.spatial_transform import LearnableSpatialTransformWrapper
from modules.squeeze_excitation import SELayer


class FFCSE_block(nn.Module):

    def __init__(self, channels, ratio_g):
        super(FFCSE_block, self).__init__()
        in_cg = int(channels * ratio_g)
        in_cl = channels - in_cg
        r = 16

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(channels, channels // r,
                               kernel_size=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv_a2l = None if in_cl == 0 else nn.Conv2d(
            channels // r, in_cl, kernel_size=1, bias=True)
        self.conv_a2g = None if in_cg == 0 else nn.Conv2d(
            channels // r, in_cg, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x

        x = id_l if type(id_g) is int else torch.cat([id_l, id_g], dim=1)
        x = self.avgpool(x)
        x = self.relu1(self.conv1(x))

        x_l = 0 if self.conv_a2l is None else id_l * self.sigmoid(self.conv_a2l(x))
        x_g = 0 if self.conv_a2g is None else id_g * self.sigmoid(self.conv_a2g(x))
        return x_l, x_g


class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups

        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels + (2 if spectral_pos_encoding else 0),
                                          out_channels=out_channels,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

        # squeeze and excitation block
        self.use_se = use_se
        if use_se:
            if se_kwargs is None:
                se_kwargs = {}
            self.se = SELayer(self.conv_layer.in_channels, **se_kwargs)

        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]

        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode,
                              align_corners=False)

        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted = x
        # ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        # ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        # ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        # ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        # if self.spectral_pos_encoding:
        #     height, width = ffted.shape[-2:]
        #     coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
        #     coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
        #     ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        # if self.use_se:
        #     ffted = self.se(ffted)

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        output = self.relu(self.bn(ffted))

        # ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
        #     0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        # ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        # ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        # output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)

        return output


class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, **fu_kwargs):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = FourierUnit(out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1), dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output


class CrossAttentionBlock(nn.Module):  # 60K params with 128,128,128
    def __init__(self, q_in_channels, kv_in_channels, channels, out_channels, num_heads=1):
        super(CrossAttentionBlock, self).__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.num_heads = num_heads
        self.q = nn.Conv1d(q_in_channels, channels, kernel_size=1)
        self.k = nn.Conv1d(kv_in_channels, channels, kernel_size=1)
        self.v = nn.Conv1d(kv_in_channels, channels, kernel_size=1)
        self.proj_out = nn.Conv1d(channels, out_channels, kernel_size=1)
        # todo norm???

    def forward(self, query, key, value):
        """

        :param query: feature map with shape (batch, c, h, w)
        :param key: feature map with shape (batch, c, h, w)
        :param value: feature map with shape (batch, c, h, w)
        :return: feature map with shape (batch, c, h, w)
        """

        b, c_q, *spatial_size = query.shape
        _, c_k, *_ = key.shape
        _, c_v, *_ = value.shape

        query = query.reshape(b, c_q, -1)
        key = key.reshape(b, c_k, -1)
        value = value.reshape(b, c_v, -1)

        q = self.q(query).reshape(b * self.num_heads, -1, query.shape[-1])
        k = self.k(key).reshape(b * self.num_heads, -1, key.shape[-1])
        v = self.v(value).reshape(b * self.num_heads, -1, value.shape[-1])

        scale = 1. / math.sqrt(math.prod(spatial_size))
        attention_weights = torch.bmm(q, k.transpose(1, 2)) * scale
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention = torch.bmm(attention_weights, v)

        attention_output = attention.reshape(b, -1, attention.shape[-1])
        attention_output = self.proj_out(attention_output)
        attention_output = attention_output.reshape(b, -1, *spatial_size)

        return attention_output


class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, use_convolutions=False, cross_attention='none',
                 cross_attention_args=None, **spectral_kwargs):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        # groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        # groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        conv2d_kwargs = dict(
            kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias, padding_mode=padding_type
        )

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, **conv2d_kwargs)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, **conv2d_kwargs)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, **conv2d_kwargs)

        if use_convolutions:
            module = nn.Identity if in_cg == 0 or out_cg == 0 else nn.Conv2d
            self.convg2g = module(in_cg, out_cg, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
            self.convg2g = module(in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu,
                                  **spectral_kwargs)

        self.cross_attention = cross_attention
        if cross_attention in ['cross_local', 'cross']:
            self.lg_cross_attention = CrossAttentionBlock(
                q_in_channels=out_cl,
                kv_in_channels=out_cl,
                channels=out_cl // cross_attention_args.get('attention_channel_scale_factor', 1),
                out_channels=out_cl,
                num_heads=cross_attention_args.get('num_heads', 1))
        if cross_attention in ['cross_global', 'cross']:
            self.gl_cross_attention = CrossAttentionBlock(
                q_in_channels=out_cg,
                kv_in_channels=out_cg,
                channels=out_cg // cross_attention_args.get('attention_channel_scale_factor', 1),
                out_channels=out_cg,
                num_heads=cross_attention_args.get('num_heads', 1))

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            cl2l = self.convl2l(x_l)
            cg2l = self.convg2l(x_g) * g2l_gate

            if self.cross_attention in ['cross_local', 'cross']:
                out_xl = self.lg_cross_attention(cl2l, cg2l, cg2l)
            else:
                out_xl = cl2l + cg2l
        if self.ratio_gout != 0:
            cl2g = self.convl2g(x_l) * l2g_gate
            cg2g = self.convg2g(x_g)

            if self.cross_attention in ['cross_global', 'cross']:
                out_xg = self.gl_cross_attention(cl2g, cg2g, cg2g)
            else:
                out_xg = cl2g + cg2g

        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect', use_convolutions=False, cross_attention='none',
                 cross_attention_args=None, enable_lfu=True, **kwargs):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, use_convolutions=use_convolutions,
                       cross_attention=cross_attention, cross_attention_args=cross_attention_args, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class FFCResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 spatial_transform_kwargs=None, inline=False, use_convolutions=False,
                 cross_attention='none', cross_attention_args=None, **conv_kwargs):
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation, norm_layer=norm_layer,
                                activation_layer=activation_layer, padding_type=padding_type,
                                use_convolutions=use_convolutions, cross_attention=cross_attention,
                                cross_attention_args=cross_attention_args, **conv_kwargs)
        self.conv2 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation, norm_layer=norm_layer,
                                activation_layer=activation_layer, use_convolutions=use_convolutions,
                                padding_type=padding_type, cross_attention=cross_attention,
                                cross_attention_args=cross_attention_args, **conv_kwargs)
        if spatial_transform_kwargs is not None:
            self.conv1 = LearnableSpatialTransformWrapper(self.conv1, **spatial_transform_kwargs)
            self.conv2 = LearnableSpatialTransformWrapper(self.conv2, **spatial_transform_kwargs)
        self.inline = inline

    def forward(self, x):
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g

        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))

        x_l, x_g = id_l + x_l, id_g + x_g  # TODO test that this is correct
        out = x_l, x_g
        if self.inline:
            out = torch.cat(out, dim=1)
        return out


class ConcatTupleLayer(nn.Module):
    def forward(self, x):
        assert isinstance(x, tuple)
        x_l, x_g = x
        assert torch.is_tensor(x_l) or torch.is_tensor(x_g)
        if not torch.is_tensor(x_g):
            return x_l
        return torch.cat(x, dim=1)


class LaMa(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', activation_layer=nn.ReLU,
                 up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                 init_conv_kwargs={}, downsample_conv_kwargs={}, resnet_conv_kwargs={},
                 spatial_transform_layers=None, spatial_transform_kwargs={},
                 add_out_act=True, max_features=1024, out_ffc=False, out_ffc_kwargs={}, use_convolutions=True,
                 cross_attention='none', cross_attention_args=None, skip_connections='none', unet_layers=0):
        assert (n_blocks >= 0)
        super().__init__()

        unet_layers_kwargs = dict(kernel_size=3, stride=1, padding=1)

        self.reflect = nn.ReflectionPad2d(3)
        down_sampling_out_channels = [ngf]
        self.skip_connections = skip_connections
        layer = [FFC_BN_ACT(input_nc, down_sampling_out_channels[-1],
                            kernel_size=7, padding=0, norm_layer=norm_layer,
                            activation_layer=activation_layer, use_convolutions=use_convolutions,
                            cross_attention='none', cross_attention_args=None, **init_conv_kwargs)]

        for _ in range(unet_layers):
            channels = down_sampling_out_channels[-1]
            layer.append(FFC_BN_ACT(channels, channels, norm_layer=norm_layer, activation_layer=activation_layer,
                                  **init_conv_kwargs, **unet_layers_kwargs))

        self.down_sampling_layers = [nn.Sequential(*layer)]

        self.resnet_layers = []
        self.up_sampling_layers = []

        # Down-sample
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            down_sampling_out_channels.append(min(max_features, ngf * mult * 2))
            layer = [FFC_BN_ACT(min(max_features, ngf * mult),
                                down_sampling_out_channels[-1],
                                kernel_size=3, stride=2, padding=1,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                use_convolutions=use_convolutions,
                                cross_attention='none',
                                cross_attention_args=None,
                                **cur_conv_kwargs)]
            if i < n_downsampling - 1:
                for j in range(unet_layers):
                    channels = down_sampling_out_channels[-1]
                    layer.append(FFC_BN_ACT(channels, channels, norm_layer=norm_layer, activation_layer=activation_layer,
                                            **cur_conv_kwargs, **unet_layers_kwargs))
            self.down_sampling_layers += [nn.Sequential(*layer)]

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        # ResNet Blocks
        for i in range(n_blocks):
            cur_resblock = FFCResnetBlock(feats_num_bottleneck, padding_type=padding_type,
                                          activation_layer=activation_layer,
                                          norm_layer=norm_layer, use_convolutions=use_convolutions,
                                          cross_attention=cross_attention,
                                          cross_attention_args=cross_attention_args,
                                          **resnet_conv_kwargs)
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
            self.resnet_layers += [cur_resblock]

        self.resnet_layers += [ConcatTupleLayer()]

        # Up-sample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)

            input_channels_num = min(max_features, ngf * mult)
            input_channels_num += down_sampling_out_channels.pop() if skip_connections == 'cat' else 0
            layer = [nn.ConvTranspose2d(input_channels_num,
                                        min(max_features, int(ngf * mult / 2)),
                                        kernel_size=3, stride=2, padding=1, output_padding=1),
                     up_norm_layer(min(max_features, int(ngf * mult / 2))),
                     up_activation
                     ]
            for _ in range(unet_layers):
                channels = min(max_features, int(ngf * mult / 2))
                layer.append(nn.Conv2d(channels, channels, **unet_layers_kwargs))
                layer.append(nn.BatchNorm2d(channels))
                layer.append(nn.ReLU())
            self.up_sampling_layers.append(nn.Sequential(*layer))

        if out_ffc:
            raise NotImplementedError
            # layer = nn.Sequential(FFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer,
            #                          norm_layer=norm_layer, inline=True, use_convolutions=use_convolutions,
            #                          cross_attention='none', cross_attention_args=None,
            #                          **out_ffc_kwargs))

        input_channels_num = ngf
        input_channels_num += down_sampling_out_channels.pop() if skip_connections == 'cat' else 0
        tmp_out = output_nc if unet_layers == 0 else input_channels_num
        layer = [nn.ReflectionPad2d(3), nn.Conv2d(input_channels_num, tmp_out, kernel_size=7, padding=0)]
        for j in range(unet_layers):
            tmp_out = output_nc if j == unet_layers - 1 else input_channels_num
            layer.append(nn.BatchNorm2d(input_channels_num))
            layer.append(nn.ReLU())
            layer.append(nn.Conv2d(input_channels_num, tmp_out, kernel_size=3, stride=1, padding=1))
        self.up_sampling_layers.append(nn.Sequential(*layer))

        self.final_act = get_activation('tanh' if add_out_act is True else add_out_act)
        self.resnet_layers = nn.Sequential(*self.resnet_layers)
        self.down_sampling_layers = nn.Sequential(*self.down_sampling_layers)
        self.up_sampling_layers = nn.Sequential(*self.up_sampling_layers)

    def forward(self, input):
        input = self.reflect(input)
        intermediate_outputs = []
        for down_layer in self.down_sampling_layers:
            input = down_layer(input)
            if self.skip_connections != 'none':
                tmp_input = input if isinstance(input[1], torch.Tensor) else (input[0],)
                intermediate_outputs.append(torch.cat(tmp_input, dim=1))
        input = self.resnet_layers(input)
        for up_layer in self.up_sampling_layers:
            if self.skip_connections != 'none':
                intermediate_output = intermediate_outputs.pop()
                if self.skip_connections == 'cat':
                    input = torch.cat([input, intermediate_output], dim=1)
                elif self.skip_connections == 'add':
                    input = input + intermediate_output
            input = up_layer(input)
        return self.final_act(input)


