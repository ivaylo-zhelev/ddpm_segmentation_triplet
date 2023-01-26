from torchmetrics.functional.image import structural_similarity_index_measure
import torch.nn.functional as F
import torch


F1_METRIC = "F1"
IOU_METRIC = "IoU"
DICE_METRIC = "Dice"
SSIM_METRIC = "SSIM"
MAE_METRIC = "MAE"


def iou(predicted, ground_truth, threshold=0.5):
    predicted = predicted.mean(1)
    ground_truth = ground_truth.mean(1)
    predicted = predicted >= threshold
    ground_truth = ground_truth >= threshold

    intersection = predicted.logical_and(ground_truth).sum()
    union = predicted.logical_or(ground_truth).sum()
    return (intersection / union).item()


def get_confusion_matrix(predicted, ground_truth, threshold=0.5):
    predicted = predicted >= threshold
    ground_truth = ground_truth >= threshold

    predicted_negation = torch.logical_not(predicted)
    ground_truth_negation = torch.logical_not(ground_truth)

    true_positives = torch.logical_and(predicted, ground_truth).sum()
    true_negatives = torch.logical_and(predicted_negation, ground_truth_negation).sum()
    false_positives = torch.logical_and(predicted, ground_truth_negation).sum()
    false_negatives = torch.logical_and(predicted_negation, ground_truth).sum()

    return true_positives, true_negatives, false_positives, false_negatives


def dice(predicted, ground_truth, threshold=0.5, eps=1e-06):
    true_positives, true_negatives, false_positives, false_negatives = get_confusion_matrix(predicted, ground_truth,
                                                                                            threshold=threshold)

    dice_score = (2 * true_positives + eps) / (2 * true_positives + false_positives + false_negatives + eps)
    return dice_score.item()


def f1_score(predicted, ground_truth, threshold=0.5, eps=1e-06):
    true_positives, true_negatives, false_positives, false_negatives = get_confusion_matrix(predicted, ground_truth,
                                                                                            threshold=threshold)

    
    precision = (true_positives + eps) / (true_positives + false_positives + eps)
    recall = (true_positives + eps) / (true_positives + false_negatives + eps)

    f1 = 2 * (precision * recall) / (precision + recall)

    return f1.item()


def mae(predicted, ground_truth):
    return F.l1_loss(predicted, ground_truth).item()


def ssim(predicted, ground_truth, *args, **kwargs):
    return structural_similarity_index_measure(predicted, ground_truth).item()


EVAL_FUNCTIONS = {
    F1_METRIC: f1_score,
    IOU_METRIC: iou,
    DICE_METRIC: dice,
    SSIM_METRIC: ssim,
    MAE_METRIC: mae
}
