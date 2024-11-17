import torch
import torch.nn as nn
from torch.nn import L1Loss, BCELoss

# *******************************************************
# NOTE : y = generated , x = ground truth translated data
# *******************************************************


def frobenius_squared_norm(a, b):
    diff = a - b
    frobenius_norm = torch.linalg.norm(diff, 'fro')
    res = frobenius_norm ** 2
    return res


def convert1channelto3channel(img, device):
    b, c, h, w = img.shape
    new_img = torch.zeros((b, 3, h, w)).to(device)
    for i in range(b):
        new_img[i, 0] = img[0, 0]
        new_img[i, 1] = img[0, 0]
        new_img[i, 2] = img[0, 0]
    return new_img


def get_features(image, model, device):
    # convert grayscale to 3 channel
    image = convert1channelto3channel(image, device)

    layers = {'0': 'conv1_1',
              '5':  'conv2_1',
              '10': 'conv3_1',
              '19': 'conv4_1',
              '28': 'conv5_1'}

    features = []
    x = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features.append(x)

    return features


def gram_matrix(tensor):
    gram = None
    d, h, w = tensor.size()
    tensor = tensor.view(d, h*w)
    gram = torch.mm(tensor, tensor.t())
    return gram/(h*w*d)


class CGANGeneratorLoss(nn.Module):
    def __init__(self):
        super(CGANGeneratorLoss, self).__init__()
        self.bce = BCELoss()
        self.l1 = L1Loss()

    def forward(self, y_patch, x, y):
        # return torch.mean(-1 * torch.log10(y_patch))
        valid = torch.ones_like(y_patch)
        loss = self.bce(y_patch, valid) + self.alpha * self.l1(y, x)
        return loss


class CGANDiscreminatorLoss(nn.Module):
    def __init__(self):
        super(CGANDiscreminatorLoss, self).__init__()
        self.bce = BCELoss()

    def forward(self, x, y, x_patch, y_patch, lambdaL1):
        # cgan_loss = torch.mean(-1*torch.log10(x_patch)) + \
        #     torch.mean(-1*torch.log10(1-y_patch))
        # # l1_loss = lambdaL1 * L1Loss(x, y, reduction="mean")
        # l1_loss = lambdaL1 * (torch.mean(torch.abs(x-y)))
        # return cgan_loss+l1_loss
        valid = torch.ones_like(y_patch)
        fake = torch.zeros_like(y_patch)
        real_loss = self.bce(x_patch, valid)
        fake_loss = self.bce(y_patch, fake)
        loss = (real_loss + fake_loss) / 2
        return loss


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.l1 = L1Loss()

    def forward(self, x_fmaps, y_fmaps, lambda0, lambda_perceptual_layer):
        ploss = 0
        p0 = 0
        p1 = 0
        # shape is fmap , batch , c , h , w
        for i in range(len(x_fmaps[0])):
            d = x_fmaps[0][i].shape[0]
            h = x_fmaps[0][i].shape[1]
            w = x_fmaps[0][i].shape[2]
            p0 = (lambda_perceptual_layer[0]/(d*h*w)) * \
                self.l1(x_fmaps[0][i], y_fmaps[0][i])
            # p0 = (lambda_perceptual_layer[0]/(d*h*w)) * \
            #     torch.mean(torch.abs(x_fmaps[0][i] - y_fmaps[0][i]))
        for i in range(len(x_fmaps[1])):
            d = x_fmaps[1][i].shape[1]
            h = x_fmaps[1][i].shape[1]
            w = x_fmaps[1][i].shape[2]
            p1 = (lambda_perceptual_layer[1]/(d*h*w)) * \
                self.l1(x_fmaps[1][i], y_fmaps[1][i])
            # p1 = (lambda_perceptual_layer[1]/(d*h*w)) * \
            #     torch.mean(torch.abs(x_fmaps[1][i] - y_fmaps[1][i]))
        ploss = lambda0*(p0+p1)
        return (ploss)


class StyleTransferLoss(nn.Module):
    def __init__(self, device):
        super(StyleTransferLoss, self).__init__()
        self.device = device

    def forward(self, x, y, model, lambda_content, lambda_style, lambda_content_layers, lambda_style_layers):
        if lambda_content_layers == None:
            lambda_content_layers = [1, 1, 1, 1, 1]
        if lambda_style_layers == None:
            lambda_style_layers = [1, 1, 1, 1, 1]
        x_fmap = get_features(x, model, self.device)
        y_fmap = get_features(y, model, self.device)

        # content loss
        content_loss = 0
        # shape is fmap , batch , c , h , w
        for i in range(5):
            b, d, h, w = x_fmap[i].shape
            for j in range(b):
                content_loss += (lambda_content_layers[i]/(d*h*w)) * \
                    frobenius_squared_norm(x_fmap[i][j][0], y_fmap[i][j][0])
        content_loss = lambda_content*content_loss

        # sytle loss
        style_loss = 0
        # shape is fmap , batch , c , h , w
        for i in range(5):
            b, d, h, w = x_fmap[i].shape
            for j in range(b):
                style_loss += (lambda_style_layers[i]/(4*(d**2))) * \
                    frobenius_squared_norm(gram_matrix(
                        x_fmap[i][j]), gram_matrix(y_fmap[i][j]))

        style_loss = lambda_style*style_loss

        return content_loss, style_loss
