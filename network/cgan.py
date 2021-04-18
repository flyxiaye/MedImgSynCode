# import paddle.fluid as fluid
# from paddle.fluid.dygraph import Conv2D, Linear, Dropout, BatchNorm, Pool2D, Conv2DTranspose, InstanceNorm, SpectralNorm, Conv3D, Conv3DTranspose
import paddle
from paddle.nn import Conv3D, BatchNorm3D, Conv3DTranspose, MaxPool3D
import paddle.nn.functional  as F
import numpy as np

class ConvBNLayer(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same'):
        super(ConvBNLayer, self).__init__()

        self.conv = Conv3D(in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding)
        self.bn = BatchNorm3D(out_channels)
    
    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = F.leaky_relu(y, 0.2)
        return y

class UpConvBNLayer(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(UpConvBNLayer, self).__init__()

        self.conv = Conv3DTranspose(in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding)
        self.bn = BatchNorm3D(out_channels)
    
    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = F.relu(y)
        return y

class Gen(paddle.nn.Layer):
    def __init__(self, in_channels=1):
        super(Gen, self).__init__()

        self.conv1 = Conv3D(in_channels, 64, 4, 2, padding='same')
        self.convbn2 = ConvBNLayer(64, 128, 4, 2)
        self.convbn3 = ConvBNLayer(128, 256, 4, 2)
        self.convbn4 = ConvBNLayer(256, 512, 4, 2)
        self.convbn5 = ConvBNLayer(512, 512, 4, 2)
        self.convbn6 = ConvBNLayer(512, 512, 4, 2)
        self.convbn7 = ConvBNLayer(512, 512, 4, 2)

        self.upconvbn1 = UpConvBNLayer(512, 512, 4, 2)
        self.upconvbn2 = UpConvBNLayer(1024, 512, 4, 2)
        self.upconvbn3 = UpConvBNLayer(1024, 512, 4, 2)
        self.upconvbn4 = UpConvBNLayer(1024, 512, 4, 2)
        self.upconvbn5 = UpConvBNLayer(256*3, 256, 4, 2)
        self.upconvbn6 = UpConvBNLayer(128*3, 128, 4, 2)
        self.upconvbn7 = UpConvBNLayer(64*3, 128, 4, 2)
        self.conv2 = Conv3D(128, 1, 1, 1)
    
    def forward(self, x):
        x = self.conv1(x)
        x1 = F.leaky_relu(x, 0.2)
        x2 = self.convbn2(x1)
        x3 = self.convbn3(x2)
        x4 = self.convbn4(x3)
        x5 = self.convbn5(x4)
        x6 = self.convbn6(x5)
        x = self.convbn7(x6)
        x = self.upconvbn1(x)
        x = paddle.concat((x, x6), axis=1)
        x = self.upconvbn2(x)
        x = paddle.concat((x, x5), axis=1)
        x = self.upconvbn3(x)
        x = paddle.concat((x, x4), axis=1)
        x = self.upconvbn4(x)
        x = paddle.concat((x, x3), axis=1)
        x = self.upconvbn5(x)
        x = paddle.concat((x, x2), axis=1)
        x = self.upconvbn6(x)
        x = paddle.concat((x, x1), axis=1)
        x = self.upconvbn7(x)
        x = self.conv2(x)

        return x


class Dsc(paddle.nn.Layer):
    def __init__(self, in_channels=2):
        super(Dsc, self).__init__()

        self.conv = Conv3D(in_channels, 64, 4, 2, padding='same')
        self.convbn1 = ConvBNLayer(64, 128, 4, 2)
        self.convbn2 = ConvBNLayer(128, 256, 4, 2)
        self.convbn3 = ConvBNLayer(256, 512, 4, 2)
        self.conv2 = Conv3D(512, 1, 1, 1)
    
    def forward(self, x):
        y = self.conv(x)
        y = F.leaky_relu(y, 0.2)
        y = self.convbn1(y)
        y = self.convbn2(y)
        y = self.convbn3(y)
        y = self.conv2(y)
        y = F.sigmoid(y)
        return y

if __name__ == '__main__':
    # with fluid.dygraph.guard(fluid.CUDAPlace(0)):
    # network = UNet3D()
    # network = ConvBNLayer(1, 32, 3)
    # network = Dsc()
    network = Gen()
    img = np.ones([1, 1, 128, 128, 128]).astype('float32')
    img = paddle.to_tensor(img)
    outs = network(img).numpy()
    
    # print(network.state_dict().keys())   ###打印网络的层名称列表
    # print(network.state_dict())
    # print(outs,outs.shape)
    print(outs.shape)