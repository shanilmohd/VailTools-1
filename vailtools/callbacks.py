import keras.backend as K
import numpy as np
from keras.callbacks import Callback


def cyclic_lr_schedule(lr0=0.2, total_steps=400, cycles=8):
    """
    Implements the cyclic learning rate schedule described in https://arxiv.org/abs/1704.00109

    Args:
        lr0: (float) Maximum learning rate, attained at the start of each cycle
        total_steps: (int) Number of learning rate updates
        cycles: (int) Number of cycles to perform over the course of total_steps learning rate updates

    Returns: (Callable[[int, float], float])
    """

    def cyclic_lr_schedule_(step, lr=0.):
        return 0.5 * lr0 * (np.cos(np.pi * (step % np.ceil(total_steps / cycles)) / np.ceil(total_steps / cycles)) + 1)

    return cyclic_lr_schedule_


class CyclicLRScheduler(Callback):
    """Cyclic learning rate scheduler.
    Args:
        schedule: (Callable[[int, float], float]) Takes an epoch index (0 indexed), current learning rate and returns a new learning rate.
        verbose: (int) 0 -> quiet, 1 -> update messages.
    """

    def __init__(self, schedule=None, lr0=0.2, total_steps=400, cycles=8, verbose=0):
        super().__init__()

        if schedule is None:
            self.schedule = cyclic_lr_schedule(
                lr0=lr0,
                total_steps=total_steps,
                cycles=cycles
            )
        else:
            self.schedule = schedule
        self.verbose = verbose
        self.step = 0

    def on_batch_begin(self, batch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        try:  # new API
            lr = self.schedule(self.step,
                               lr=float(K.get_value(self.model.optimizer.lr)))
        except TypeError:  # compatibility with old API
            lr = self.schedule(self.step)

        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function should be float.')

        K.set_value(self.model.optimizer.lr, lr)

        if self.verbose > 0:
            with open('lr_schedule.log', 'a') as f:
                print(f'\nStep {self.step}: learning rate = {lr}.', file=f)
        self.step += 1
