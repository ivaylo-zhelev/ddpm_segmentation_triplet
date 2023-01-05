import torch.nn.functional as F
import torch

def mse(anchor, positive, reduction,
        *args, **kwargs):
    return F.mse_loss(anchor, positive, reduction=reduction)

def triplet_margin_loss(anchor, positive, negative, p=2.0, eps=1e-6, margin=1.0, reduction='none',
                              *args, **kwargs):
    return F.triplet_margin_loss(anchor, positive, negative,
                                 p=p, eps=eps, margin=margin, reduction=reduction)


def exact_triplet_margin_loss(anchor, positive, negative, p=2.0, eps=1e-6, margin=None, reduction='none',
                              *args, **kwargs):
    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.size() == negative.size(), "Input sizes between anchor and negative must be equal."
    assert positive.size() == negative.size(), "Input sizes between positive and negative must be equal."

    d_p = F.pairwise_distance(anchor, positive, p, eps)
    d_n = F.pairwise_distance(anchor, negative, p, eps)

    if margin is None:
        margin = F.pairwise_distance(positive, negative, p, eps)

    dist_hinge = (margin + d_p - d_n) ** 2
    if reduction == 'mean':
        loss = torch.mean(dist_hinge)
    elif reduction == 'sum':
        loss = torch.sum(dist_hinge)
    else:
        loss = dist_hinge

    return loss


def triplet_loss_dynamic_margin(anchor, positive, negative, p=2.0, eps=1e-6, reduction='none',
                                *args, **kwargs):
    return exact_triplet_margin_loss(anchor, positive, negative, p=p, eps=eps, reduction=reduction, margin=None)


def regularized_triplet_loss(anchor, positive, negative,
                             p=2.0, eps=1e-6, margin=1.0,
                             regularization_margin=10.0,
                             reduction='none',
                             regularize_to_white_image = True,
                             *args, **kwargs):
    loss = F.triplet_margin_loss(anchor, positive, negative, margin=margin, p=p, eps=eps, reduction=reduction)
    black_image = torch.zeros_like(anchor)
    regularization = F.triplet_margin_loss(anchor, positive, black_image, margin=regularization_margin, p=p, eps=eps, reduction=reduction)
    
    if regularize_to_white_image:
        white_image = torch.ones_like(anchor)
        regularization *= F.triplet_margin_loss(anchor, positive, white_image, margin=regularization_margin, p=p, eps=eps, reduction=reduction)

    return loss * regularization
