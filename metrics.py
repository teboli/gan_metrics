import torch
from torch import nn
from torchvision import models, transforms

import losses, networks


class InceptionScore(nn.Module):
    def __init__(self, state_dict=None, batch_size=100, splits=10):
        super(InceptionScore, self).__init__()
        self.batch_size = batch_size
        self.splits = splits
        self.model = models.inception_v3(pretrained=True)  # model pretrained on ImageNet
        if state_dict is not None:
            self.model.load_state_dict(state_dict)

    def forward(self, input):
        # Inception works only with 3x299x299 images. You can crop (like center crop or 10 crops)
        # the images or rescale them before feeding them into Inception.
        assert(input.shape[1] == 3 and input.shape[2] == 299 and input.shape[3] == 299)
        self.model.eval()

        n = input.shape
        n_batches = int(math.ceil(float(n) / float(self.batch_size)))
        probs = []
        for i in range(n_batches):
            start = i * self.batch_size
            end   = min(i+1 * self.batch_size, n)
            input_split = input[start:end]
            probs.append(self.model(input_split).softmax(dim=1))
        probs = torch.cat(probs, dim=0)

        scores = []
        for i in range(self.splits):
            start = i     * n // self.splits
            end   = (i+1) * n // self.splits
            probs_split = probs[start:end]
            p = p.mean(dim=0,keepdim=True).log()
            kl = torch.nn.functional.kl_div(probs_split, p, reduction='none').sum(dim=1).mean()
            scores.append(kl.exp().item())

        return np.mean(scores), np.std(scores)


class FCNScore(nn.Module):
    def __init__(self, state_dict=None):
        self.model = networks.fcn_8s()
        if state_dict is not None:
            self.model.load_state_dict(state_dict)

    def forward(self, input, target):
        self.model.eval()
        input_feat  = self.model(input).softmax(dim=1)
        target_feat = self.model(target).softmax(dim=1)
        return torch.nn.functional.kl_div(input_feat, target_feat).exp()


if __name__ == '__main__':
    # Toy example
    a = torch.rand(1,3,299,299)
    b = torch.rand(1,3,299,299)

    # For a classical RGB image, don't forget to center the data with those means and stds.
    # normalize = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    # a = normalize(a)
    # b = normalize(b)

    fcn_criterion = FCNScore()
    inception_criterion  = InceptionScore()

    perceptual_loss = fcn_criterion(a,b)
    inception_score = inception_criterion(a,b)
