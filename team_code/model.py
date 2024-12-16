import torch
from torch import nn
from torch.nn.functional import normalize
from resnet import resnet18, resnet34


class MLP(nn.Module):
    def __init__(self, *args):
        super().__init__()
        nets = []
        for l in args:
            if l == 'A':
                layer = nn.ReLU(inplace=True)
            elif l == 'D':
                layer = nn.Dropout2d()
            else:
                layer = nn.Linear(l[0], l[1])
            nets.append(layer)
        self.net = nn.Sequential(*nets)

    def forward(self, x):
        return self.net(x)

class QuadNet(nn.Module):
    def __init__(self, dim_in, dim_out = None, normalize=True):
        super().__init__()
        if dim_out is None:
            dim_out = dim_in
        self.norm = nn.LayerNorm([dim_in]) if normalize else nn.Sequential()
        self.x_ = nn.Linear(dim_in, dim_in)
        self.FC = nn.Linear(dim_in, dim_out)

    def forward(self, x):  # x:(B, dim_in)
        x = self.norm(x)
        x_ = self.x_(x)
        out = x * x_
        out = self.FC(out)
        return out

class Generator(nn.Module):
    def __init__(self, nz=512, ngf=64, nc=1):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

class Net(nn.Module):
    def __init__(self, pretrain=False):
        super().__init__()
        self.cnn_front = resnet34(pretrained=pretrain)
        self.cnn_front.fc = nn.Linear(512, 512)
        self.cnn_left = resnet34(pretrained=pretrain)
        self.cnn_left.fc = nn.Linear(512, 512)
        self.cnn_right = resnet34(pretrained=pretrain)
        self.cnn_right.fc = nn.Linear(512, 512)

        self.measurements = MLP((1 + 2 + 6, 128), 'A', (128, 128), 'A')
        self.join = MLP((512*4 + 128, 512), 'A', (512, 512))
        self.F      = QuadNet(512)
        self.dF     = QuadNet(512)
        self.d2F    = QuadNet(512)
        self.d3F    = QuadNet(512)
        self.out1   = Generator()  # lidar reconstruct
        self.out2   = MLP((512, 512), 'A', (512, 256), 'A', (256, 3))  # steer, dx, dy
        self.out3   = MLP((512, 512), 'A', (512, 256), 'A', (256, 1))  # acc(throttle-brake)

    def forward(self, img_front, img_left, img_right, last_img_front, state):
        front, _ = self.cnn_front(torch.cat((img_front, last_img_front), 0))
        front_feature, last_front_feature = front.chunk(2)
        left_feature, _ = self.cnn_left(img_left)
        right_feature, _ = self.cnn_right(img_right)

        m_feature = self.measurements(state)
        feature = self.join(torch.cat([front_feature, left_feature, right_feature,
                                       last_front_feature, m_feature], 1))
        F = self.F(feature)
        dF = self.dF(F)
        d2F = self.d2F(dF)
        d3F = self.d3F(d2F)

        lidar0 = self.out1(F[..., None, None]).squeeze()
        lidar1 = self.out1((F+dF)[..., None, None]).squeeze()
        steer0 = self.out2(dF)
        steer1 = self.out2(dF+d2F)
        acc0 = self.out3(d2F)
        acc1 = self.out3(d2F+d3F)
        return lidar0, steer0, acc0, lidar1, steer1, acc1
