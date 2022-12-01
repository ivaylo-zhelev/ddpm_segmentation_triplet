from torchmetrics.functional.image structural_similarity_index_measure
from torchmetrics.functional.classification import dice


IOU_METRIC = "IoU"
DICE_METRIC = "Dice"
SSIM_METRIC = "SSIM"

def iou(predicted, ground_truth, threshold=0.5):
    predicted = predicted.mean(1)
    ground_truth = ground_truth.mean(1)
    predicted = predicted >= threshold
    ground_truth = ground_truth >= threshold

    intersection = predicted.logical_and(ground_truth).sum()
    union = predicted.logical_or(ground_truth).sum()
    return (intersection / union).item()

def dice(predicted, ground_truth, threshold=0.5):
    predicted = (predicted >= threshold).int()
    ground_truth = (ground_truth >= threshold).int()
    return dice(predicted, ground_truth).item()

def ssim(predicted, ground_truth, *args, **kwargs):
    return structural_similarity_index_measure(predicted, ground_truth).item()

EVAL_FUNCTIONS = {
    IOU_METRIC: iou,
    DICE_METRIC: dice,
    SSIM_METRIC: ssim
}
