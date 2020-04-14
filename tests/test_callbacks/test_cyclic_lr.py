import numpy as np
from tensorflow.keras import backend as K

from vailtools.callbacks import CyclicLRScheduler


class DummyModel:
    def __init__(self):
        self.optimizer = DummyOptimizer()


class DummyOptimizer:
    def __init__(self):
        self.lr = K.variable(0.0)


def test_cyclic_lr_scheduler():
    lr_schedule = CyclicLRScheduler(cycle_period=64)
    lr_schedule.model = DummyModel()
    lrs = []
    for i in range(65):
        lr_schedule.on_batch_begin(i)
        lrs.append(float(K.get_value(lr_schedule.model.optimizer.lr)))

    # Maximum LR value occurs at the start of every cycle
    assert np.argmax(lrs) == 0
    # Minimum LR value at the end of every cycle
    # New cycle starts on step 64, since cycle_period=64
    assert np.argmin(lrs) == 63
