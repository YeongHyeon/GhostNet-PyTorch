import torch
import torch.nn as nn
import torch.nn.functional as F

class Neuralnet(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        self.who_am_i = "GhostNet"

        self.dim_h = kwargs['dim_h']
        self.dim_w = kwargs['dim_w']
        self.dim_c = kwargs['dim_c']
        self.num_class = kwargs['num_class']
        self.k_size1 = kwargs['k_size1']
        self.k_size2 = kwargs['k_size2']
        self.ratio = kwargs['ratio']
        self.filters = kwargs['filters']

        self.learning_rate = kwargs['learning_rate']
        self.path_ckpt = kwargs['path_ckpt']

        self.ngpu = kwargs['ngpu']
        self.device = kwargs['device']

        self.params, self.names = [], []
        self.params.append(WarmupConv(self.dim_c, self.filters[0], self.k_size1, stride=1, name="warmup").to(self.device))
        self.names.append("warmupconv")

        for idx_filter, filter in enumerate(self.filters[:-1]):

            self.params.append(GhostBlock(self.filters[idx_filter], self.filters[idx_filter]*2, self.filters[idx_filter], \
                self.k_size1, self.k_size2, ratio=2, stride=1, name="ghost%d_1" %(idx_filter+1)).to(self.device))
            self.names.append("ghost%d_1" %(idx_filter+1))
            self.params.append(GhostBlock(self.filters[idx_filter], self.filters[idx_filter]*3, self.filters[idx_filter+1], \
                self.k_size1, self.k_size2, ratio=2, stride=2, name="ghost%d_2" %(idx_filter+1)).to(self.device))
            self.names.append("ghost%d_2" %(idx_filter+1))

        self.params.append(Classifier(self.filters[-1], self.num_class, name='classifier').to(self.device))
        self.names.append("classifier")
        self.modules = nn.ModuleList(self.params)

    def forward(self, x):

        for idx_param, _ in enumerate(self.params):
            x = self.params[idx_param](x)
        y_hat = x

        return {'y_hat':y_hat}

    def loss(self, dic):

        y, y_hat = dic['y'], dic['y_hat']
        loss_ce = nn.CrossEntropyLoss()
        opt_b = loss_ce(y_hat, target=y)
        opt = torch.mean(opt_b)

        return {'opt': opt}

class WarmupConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, name=""):
        super().__init__()
        self.warmup = nn.Sequential()
        self.warmup.add_module("%s_conv" %(name), nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2))
        self.warmup.add_module("%s_bn" %(name), nn.BatchNorm2d(out_channels))
        self.warmup.add_module("%s_act" %(name), nn.ReLU())

    def forward(self, x):

        out = self.warmup(x)
        return out

class Classifier(nn.Module):

    def __init__(self, in_channels, out_channels, name=""):
        super().__init__()
        self.clf = nn.Sequential()
        self.clf.add_module("%s_lin0" %(name), nn.Linear(in_channels, int(in_channels*1.5)))
        self.clf.add_module("%s_act0" %(name), nn.ReLU())
        self.clf.add_module("%s_lin1" %(name), nn.Linear(int(in_channels*1.5), out_channels))

    def forward(self, x):

        gap = torch.mean(x, axis=(2, 3))
        return self.clf(gap)

class GhostBlock(nn.Module):

    def __init__(self, in_channels, ext_channels, out_channels, kernel_size1=3, kernel_size2=3, ratio=2, stride=2, name=""):
        super().__init__()

        self.stride = stride

        self.ghost = nn.Sequential()
        self.ghost.add_module("%s_ghost1" %(name), GhostModule(in_channels, ext_channels, \
            kernel_size1, kernel_size2, ratio, 1, True, name="%s_ghost_b1" %(name)))
        if(self.stride == 2):
            self.ghost.add_module("%s_conv_dw" %(name), nn.Conv2d(ext_channels, ext_channels, \
                kernel_size1, stride, groups=ext_channels, padding=kernel_size1//2))
            self.ghost.add_module("%s_bn_dw" %(name), nn.BatchNorm2d(ext_channels))
        self.ghost.add_module("%s_ghost2" %(name), GhostModule(ext_channels, out_channels, \
            kernel_size1, kernel_size2, ratio, 1, False, name="%s_ghost_b2" %(name)))

        if(self.stride == 2):
            self.shortcut = nn.Sequential()
            self.shortcut.add_module("%s_conv_resi" %(name), nn.Conv2d(in_channels, out_channels, \
                kernel_size1, stride, groups=1, padding=kernel_size1//2))
            self.shortcut.add_module("%s_bn_resi" %(name), nn.BatchNorm2d(out_channels))

    def forward(self, x):

        out = self.ghost(x)
        if(self.stride == 1): resi = x
        else: resi = self.shortcut(x)
        return resi + out

class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size1=3, kernel_size2=3, ratio=2, stride=1, activation=True, name=""):
        super().__init__()

        self.out_channels = out_channels
        group_channels = int(out_channels / ratio)
        new_channels = group_channels*(ratio-1)

        self.conv_prime = nn.Sequential()
        self.conv_prime.add_module("%s_conv_p" %(name), nn.Conv2d(in_channels, group_channels, \
            kernel_size1, stride, padding=kernel_size1//2)) # primary convolution (ordinary)
        self.conv_prime.add_module("%s_bn_p" %(name), nn.BatchNorm2d(group_channels))
        self.conv_prime.add_module("%s_act_p" %(name), nn.ReLU())

        self.conv_cheap = nn.Sequential()
        self.conv_cheap.add_module("%s_conv_c" %(name), nn.Conv2d(group_channels, new_channels, \
            kernel_size2, 1, groups=group_channels, padding=kernel_size2//2)) # primary convolution (ordinary)
        self.conv_cheap.add_module("%s_bn_c" %(name), nn.BatchNorm2d(new_channels))
        self.conv_cheap.add_module("%s_act_c" %(name), nn.ReLU())

    def forward(self, x):
        x_p = self.conv_prime(x)
        x_c = self.conv_cheap(x_p)
        out = torch.cat([x_p, x_c], dim=1)
        return out[:, :self.out_channels, :, :]
