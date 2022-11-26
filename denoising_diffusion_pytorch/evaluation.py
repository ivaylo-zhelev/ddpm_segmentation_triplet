from torchmetrics.functional import dice_score, structural_similarity_index_measure

IOU_METRIC = "IoU"
DICE_METRIC = "Dice"
SSIM_METRIC = "SSIM"

def iou(predicted, ground_truth):
    intersection = predicted.logical_and(ground_truth).sum()
    union = predicted.logical_or(ground_truth).sum()
    return intersection / union

def dice(predicted, ground_truth):
    return dice_score(predicted, ground_truth)

def ssim(predicted, ground_truth):
    return structural_similarity_index_measure(predicted, ground_truth)

EVAL_FUNCTIONS = {
    IOU_METRIC: iou,
    DICE_METRIC: dice,
    SSIM_METRIC: ssim
}
