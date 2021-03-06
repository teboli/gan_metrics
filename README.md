# gan metrics

This repository has been made to bring some quantitative metrics to help evaluating images producing with GANs. It contains ready-to-use pytorch code for the Inception score [1] and the FCN score [2].

The inception network is the one of torchvision and comes with pretrained weights in ImageNet dataset. The FCN network is pretrained on Pascal VOC 2012 dataset. Feel free to retrain one or another network on your own dataset as done in [2].

The Inception score is adapted from the Inception score Chainer implementation of hvy: https://github.com/hvy/chainer-inception-score. The FCN score code is heavily adapted from the FCN pytorch implementation of wkentaro: https://github.com/wkentaro/pytorch-fcn.

The weigths for this model, trained on Pascal VOC 2012, can be downloaded at: https://drive.google.com/uc?id=0B9P1L--7Wd2vT0FtdThWREhjNkU.

## Requirements

- [pytorch](https://github.com/pytorch/pytorch) >= 0.4.0
- [torchvision](https://github.com/pytorch/vision) >= 0.2.1

## References

[1] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec
Radford, and Xi Chen. Improved techniques for training gans. In
Advances in Neural Information Processing Systems 29, pages 2234–2242.
2016.

[2] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros.
Image-to-image translation with conditional adversarial networks.
CVPR, 2017
