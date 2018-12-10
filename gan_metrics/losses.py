import torch
import torch.nn.functional as F

def per_pixel_acc(input, num_classes):
    output = torch.zeros_like(input)
    for c in range(num_classes):
        output += input.eq(c * torch.ones_like(input)).float().div(input.numel())
    return output.div(num_classes)


def per_class_acc(input, target, num_classes):
    output = torch.zeros_like(input)
    for c in range(num_classes):
        input_c  = input.eq(c * torch.zeros_like(input)).float()
        target_c = target.eq(c * torch.zeros_like(target)).float()
        tp = input_c * target_c
        f = 1 - (input_c * target_c)
        denom = (tp + f).sum()
        denom[denom > 0] = 1
        output += tp.sum().div()
    return output.div(num_classes)


def class_iou(input, target, num_classes):
    output = torch.zeros_like(input)
    for c in range(num_classes):
        input_c  = input.eq(c * torch.zeros_like(input)).float()
        target_c = target.eq(c * torch.zeros_like(output)).float()
        inter = input_c * target_c
        union = input_c + target_c - inter
        output += inter.sum().div(union.sum())
    return output.div(num_classes)
