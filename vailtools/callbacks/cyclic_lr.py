import logging

import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback


def cyclic_lr_schedule(lr0=0.2, cycle_period=10):
    """
    Implements the cyclic learning rate schedule described in https://arxiv.org/abs/1704.00109

    Args:
        lr0: (float) Maximum learning rate, attained at the start of each cycle
        cycle_period: (int) Length of a cycle in steps (epochs or batches)

    Returns: (Callable[[int, float], float])
    """

    def cyclic_lr_schedule_(step, lr=0.0):
        return 0.5 * lr0 * (np.cos(np.pi * (step % cycle_period) / cycle_period) + 1)

    return cyclic_lr_schedule_


class CyclicLRScheduler(Callback):
    def __init__(self, cycle_period, lr0=0.2, schedule=None, verbose=0):
        """
        Args:
            cycle_period: (int)
                Number of batches that constitute a single cycle.
            lr0: (float)
                Initial learning rate.
            schedule: (Callable[[int, float], float])
                Takes an epoch index (0 indexed) and  the current learning rate.
                Returns a new learning rate.
            verbose: (int) 0 -> quiet, 1 -> update messages.
        """
        super().__init__()

        if schedule is None:
            self.schedule = cyclic_lr_schedule(lr0=lr0, cycle_period=cycle_period)
        else:
            self.schedule = schedule
        self.verbose = verbose
        self.step = 0

    def on_batch_begin(self, batch, logs=None):
        lr = self.schedule(self.step, lr=float(K.get_value(self.model.optimizer.lr)))
        K.set_value(self.model.optimizer.lr, lr)
        logging.info(f"Step {self.step}: learning rate = {lr}.")
        self.step += 1
