import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from .Flatten import Flatten


class ProtoNet(nn.Module):
    def __init__(self, **kwargs):
        super(ProtoNet, self).__init__()
        encoder = self._load_protonet_conv(**kwargs)
        self.encoder = encoder.cuda()

    def _load_protonet_conv(self, **kwargs):
        x_dim = kwargs["x_dim"]
        hid_dim = kwargs["hid_dim"]
        z_dim = kwargs["z_dim"]

        encoder = nn.Sequential(
            self._conv_block(x_dim[0], 16),
            self._conv_block(16, 32),
            self._conv_block(32, 64),
            self._conv_block(64, 128),
            self._conv_block(128, z_dim),
            # self._conv_block(x_dim[0], hid_dim),
            # self._conv_block(hid_dim, hid_dim),
            # self._conv_block(hid_dim, hid_dim),
            # self._conv_block(hid_dim, hid_dim),
            # self._conv_block(hid_dim, z_dim),
            Flatten(),
        )
        return encoder

    def _conv_block(self, in_channels, out_channels, kernels=1):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernels, padding=1
            ),  # change kernel size to 1
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1),
        )

    # edited later
    #def forward(self, x):
    #    return self.encoder(x)

    def set_forward_loss(self, sample):
        sample_images = sample["sample"].cuda()
        n_way = sample["n_way"]
        n_support = sample["n_support"]
        n_query = sample["n_query"]

        x_support = sample_images[:, :n_support]
        x_query = sample_images[:, n_support:]

        # target indices are 0 ... n_way-1
        target_inds = (
            torch.arange(0, n_way).view(n_way, 1, 1).expand(n_way, n_query, 1).long()
        )
        target_inds = Variable(target_inds, requires_grad=False)
        target_inds = target_inds.cuda()

        # encode images of the support and the query set
        x = torch.cat(
            [
                x_support.contiguous().view(n_way * n_support, *x_support.size()[2:]),
                x_query.contiguous().view(n_way * n_query, *x_query.size()[2:]),
            ],
            0,
        )
        # print('************************')
        # print(x)
        # print(x.shape)
        # print('************************')
        z = self.encoder.forward(x)
        # print(z)
        # print(z.shape)
        z_dim = z.size(-1)  # usually 64
        z_proto = z[: n_way * n_support].view(n_way, n_support, z_dim).mean(1)
        z_query = z[n_way * n_support :]

        # compute distances
        dists = self._euclidean_dist(z_query, z_proto)

        # compute probabilities
        log_p_y = F.log_softmax(-dists, dim=1).view(n_way, n_query, -1)

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            "loss": loss_val.item(),
            "acc": acc_val.item(),
            "y_hat": y_hat,
        }

    def _euclidean_dist(self, x, y):
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.pow(x - y, 2).sum(2)

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)
        
        
        
    def predict(self, support, query, n_way, n_support, n_query):
        support = support.cuda()
        # print(f"support: {support.size()}")
        x_support = support
        x_query = query

        target_inds = torch.arange(0, n_way).view(n_way, 1, 1).expand(n_way, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)
        target_inds = target_inds.cuda()

        # print(f"x_support: {x_support.size()}")
        # print(f"x_query: {x_query.size()}")
        x = torch.cat([x_support.contiguous().view(n_way * n_support, *x_support.size()[2:]), x_query.contiguous().view(n_query, *x_query.size()[2:])], 0)
        z = self.encoder.forward(x)
        # print(f"z: {z.size()}")
        z_dim = z.size(-1) #usually 64
        z_proto = z[:n_way*n_support].view(n_way, n_support, z_dim).mean(1)
        z_query = z[n_way*n_support:]

        #compute distances
        dists = self._euclidean_dist(z_query, z_proto)

        #compute probabilities
        log_p_y = F.log_softmax(-dists, dim=1).view(1, n_query, -1)

        _, y_hat = log_p_y.max(2)

        return y_hat
