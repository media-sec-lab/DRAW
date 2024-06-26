from . import *
import torch.nn.functional as F
import math


class SpatialPyramidPooling2d(nn.Module):
    def __init__(self, num_level=3, pool_type='max_pool'):
        super(SpatialPyramidPooling2d, self).__init__()
        self.num_level = num_level
        self.pool_type = pool_type
    def forward(self, x):
        N, C, H, W = x.size()
        for i in range(self.num_level):
            level = i + 1
            kernel_size = (math.ceil(H / level), math.ceil(W / level))
            stride = (math.ceil(H / level), math.ceil(W / level))
            padding = (math.floor((kernel_size[0] * level - H + 1) / 2), math.floor((kernel_size[1] * level - W + 1) / 2))
            if self.pool_type == 'max_pool':
                tensor = (F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)).view(N, -1)
            else:
                tensor = (F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)).view(N, -1)
            if i == 0:
                res = tensor
            else:
                res = torch.cat((res, tensor), 1)  #  i=1 64x（1+4） i=2  64x（1+4+9）
        # print(res.shape) # B,1920(num=4)    896(num=3)
        return res


# class SoftPool2D(torch.nn.Module):
#     def __init__(self, kernel_size, stride):
#         super(SoftPool2D,self).__init__()
#         self.kernel_size = kernel_size
#         self.stride = stride
#
#     def forward(self, x):
#         x = self.soft_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
#         return x
#
#     def soft_pool2d(self, x, kernel_size=2, stride=None):
#         # kernel_size = (kernel_size, kernel_size)
#         # if stride is None:
#         #     stride = kernel_size
#         # else:
#         #     stride = (stride, stride)
#         _, c, h, w = x.shape
#         e_x = torch.sum(torch.exp(x),axis=1,keepdim=True)
#         return F.avg_pool2d(x * e_x, kernel_size, stride=stride) * (sum(kernel_size))/(F.avg_pool2d(e_x, kernel_size, stride=stride) * (sum(kernel_size)))


# class SPPLayer(torch.nn.Module):
#     # 定义Layer需要的额外参数（除Tensor以外的）
#     def __init__(self, num_levels, pool_type='max_pool'):
#         super(SPPLayer, self).__init__()
#
#         self.num_levels = num_levels
#         self.pool_type = pool_type
#
#     # forward()的参数只能是Tensor(>=0.4.0) Variable(< 0.4.0)
#     def forward(self, x):
#         num, c, h, w = x.size()
#         level = 1
#         for i in range(self.num_levels):
#             level <<= 1
#             '''
#             The equation is explained on the following site:
#             http://www.cnblogs.com/marsggbo/p/8572846.html#autoid-0-0-0
#             '''
#             kernel_size = (math.ceil(h / level), math.ceil(w / level))  # kernel_size = (h, w)
#             padding = (
#                 math.floor((kernel_size[0] * level - h + 1) / 2), math.floor((kernel_size[1] * level - w + 1) / 2))
#             # update input data with padding
#             #  class torch.nn.ZeroPad2d(padding)[source]
#             #
#             #     Pads the input tensor boundaries with zero.
#             #
#             #     For N`d-padding, use :func:`torch.nn.functional.pad().
#             #     Parameters:	padding (int, tuple) – the size of the padding. If is int, uses the same padding in all boundaries.
#             # If a 4-tuple, uses (paddingLeft, paddingRight, paddingTop, paddingBottom)
#             zero_pad = torch.nn.ZeroPad2d((padding[1], padding[1], padding[0], padding[0]))
#             x_new = zero_pad(x)
#
#             # update kernel and stride
#             h_new, w_new = x_new.size()[2:]
#
#             kernel_size = (math.ceil(h_new / level), math.ceil(w_new / level))
#             stride = (math.floor(h_new / level), math.floor(w_new / level))
#
#             # 选择池化方式
#             if self.pool_type == 'max_pool':
#                 tensor = F.max_pool2d(x_new, kernel_size=kernel_size, stride=stride).view(num, -1)
#             elif self.pool_type == 'avg_pool':
#                 tensor = F.avg_pool2d(x_new, kernel_size=kernel_size, stride=stride).view(num, -1)
#
#             # soft_pool = SoftPool2D(kernel_size, stride)
#             # tensor = soft_pool(x_new).view(num, -1)
#             # 展开、拼接
#             if (i == 0):
#                 x_flatten = tensor.view(num, -1)
#             else:
#                 x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)
#
#         return x_flatten

class Decoder(nn.Module):
    '''
    Decode the encoded image and get message
    '''

    def __init__(self, H, W, message_length, blocks=5, channels=64):
        super(Decoder, self).__init__()

        stride_blocks = int(np.log2(H // int(np.sqrt(message_length))))
        keep_blocks = max(blocks - stride_blocks, 0)

        self.first_layers = nn.Sequential(
            ConvBNRelu(2, channels),
            SENet_decoder(channels, channels, blocks=stride_blocks + 1),
            ConvBNRelu(channels * (2 ** (stride_blocks)), channels),
        )
        self.keep_layers = SENet(channels, channels, blocks=keep_blocks)

        self.final_layer = nn.Sequential(nn.Linear(896, message_length),)
        self.ada_pool = SpatialPyramidPooling2d()

    def forward(self, noised_image):  # SPP
        # newSPP3_64channel

        x = self.first_layers(noised_image)
        x = self.keep_layers(x)
        # print(x.shape)
        x = self.ada_pool(x)
        x = self.final_layer(x)

        # # newSPP4_1channel
        # x = self.first_layers(noised_image)
        # x = self.keep_layers(x)
        # x = self.ada_pool(x)
        # x = self.final_layer(x)

        return x

    # # 2023.4.11. 11:13 4x4
    #     self.first_layers = nn.Sequential(
    #         ConvBNRelu(2, channels),
    #         SENet_decoder(channels, channels, blocks=stride_blocks + 2),
    #         ConvBNRelu(channels * (2 ** (stride_blocks+1)), channels),
    #     )
    #     self.keep_layers = SENet(channels, channels, blocks=keep_blocks)
    #     self.final_layer = nn.Sequential(nn.Linear(896, message_length),)
    #     self.ada_pool = SpatialPyramidPooling2d()
    #
    # def forward(self, noised_image):  # SPP
    #     x = self.first_layers(noised_image)
    #     x = self.keep_layers(x)
    #     x = self.ada_pool(x)
    #     x = self.final_layer(x)
    #     return x

    # 2023.4.11. 11:18  16x16
    #     self.first_layers = nn.Sequential(
    #         ConvBNRelu(2, channels),
    #         SENet_decoder(channels, channels, blocks=stride_blocks),
    #         ConvBNRelu(channels * (2 ** (stride_blocks-1)), channels),
    #     )
    #     self.keep_layers = SENet(channels, channels, blocks=keep_blocks)
    #     self.final_layer = nn.Sequential(nn.Linear(896, message_length),)
    #     self.ada_pool = SpatialPyramidPooling2d()
    #
    # def forward(self, noised_image):  # SPP
    #     x = self.first_layers(noised_image)
    #     x = self.keep_layers(x)
    #     x = self.ada_pool(x)
    #     x = self.final_layer(x)
    #     return x

    # # # 2023.4.11. 12:47 for tsm
    #     self.pre_layer = ConvBNRelu(2, channels)
    #     self.first_layers = nn.Sequential(
    #         SENet_decoder(channels, channels, blocks=stride_blocks + 1),
    #         ConvBNRelu(channels * (2 ** (stride_blocks)), channels),
    #     )
    #     self.keep_layers = SENet(channels, channels, blocks=keep_blocks)
    #     self.final_layer = nn.Sequential(nn.Linear(1792, message_length),)
    #     self.ada_pool = SpatialPyramidPooling2d()
    #
    # def forward(self, noised_image):  # SPP
    #     x = self.pre_layer(noised_image)
    #     # print(x.shape)
    #     x1 = self.keep_layers(x)
    #     # print(x1.shape)
    #
    #     x_spp1 = self.ada_pool(x1)
    #
    #     x = self.first_layers(x)
    #     # print(x.shape)
    #     x = self.keep_layers(x)
    #     # print(x.shape)
    #
    #     x_spp2 = self.ada_pool(x)
    #
    #     x_spp = torch.cat((x_spp1, x_spp2), dim=-1)
    #     # print(x_spp.shape)
    #
    #     out = self.final_layer(x_spp)
    #     return out

# SPP4
# class Decoder(nn.Module):
#     '''
#     Decode the encoded image and get message
#     '''
#
#     def __init__(self,message_length, blocks=5, channels=64):
#         super(Decoder, self).__init__()
#
#         stride_blocks = int(np.log2(128 // int(np.sqrt(message_length))))
#         keep_blocks = max(blocks - stride_blocks, 0)
#
#         self.first_layers = nn.Sequential(
#             ConvBNRelu(2, channels),
#             SENet_decoder(channels, channels, blocks=stride_blocks),
#             ConvBNRelu(channels * (2 ** (stride_blocks-1)), channels),
#         )
#         self.keep_layers = SENet(channels, channels, blocks=keep_blocks)
#
#         # self.final_layer_conv = ConvBNRelu(channels, 1)
#         self.final_layer = nn.Sequential(nn.Linear(1920, 64))
#         self.ada_pool = SpatialPyramidPooling2d()
#
#     # self.flat = nn.Flatten()
#     def forward(self, noised_image):  # SPP
#         x = self.first_layers(noised_image)
#         x = self.keep_layers(x)
#         # print(x.shape)
#         x = self.ada_pool(x)
#
#         x = self.final_layer(x)
#
#         return x

