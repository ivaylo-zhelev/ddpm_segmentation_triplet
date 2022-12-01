import torch.nn.functional as F

def mse(anchor, positive, reduction, *args, **kwargs):
    return F.mse_loss(anchor, positive, reduction=reduction)

def exact_triplet_margin_loss(anchor, positive, negative, p=2.0, eps=1e-6, margin=None, reduction='none'):
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

def regularized_triplet_loss(anchor, positive, negative, p=2.0, eps=1e-6, reduction='none', *args, **kwargs):
    return exact_triplet_margin_loss(anchor, positive, negative, p=p, eps=eps, reduction=reduction, margin=None)