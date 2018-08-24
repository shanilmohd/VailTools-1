from ..metrics import numpy_metrics


def np_iou_loss(y_true, y_pred):
    return 1 - numpy_metrics.iou_score(y_true, y_pred)