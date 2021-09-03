"""Module for lr schedulers."""
from typing import Optional

import tensorflow as tf
from typeguard import typechecked


class WarmupCosineDecayRestarts(tf.keras.optimizers.schedules.LearningRateSchedule):
    """A cosine decay schedule with restarts and warmup."""

    @typechecked
    def __init__(
        self,
        initial_learning_rate: float,
        first_decay_steps: int,
        t_mul: float = 2.0,
        m_mul: float = 1.0,
        alpha: float = 0.0,
        steps_per_epoch: Optional[int] = None,
        warmup_epochs: int = 0,
        name: Optional[str] = None,
    ):
        """Apply cosine decay with restarts and warmup to the learning rate.

        Parameters
        ----------
        initial_learning_rate : int
            The initial learning rate.
        first_decay_steps : int
            Number of steps to decay over.
        t_mul : float, optional
            Use to derive the number of iterations in the i-th period, by default `2.0`.
        m_mul : float, optional
            Use to derive the initial learning rate of the i-th period, by default `1.0`.
        alpha : float, optional
            Minimum learning rate value as a fraction of the initial_learning_rate,
            by default `0.0`.
        steps_per_epoch : int, optional
            Number of steps per epoch, by default `None`.
        warmup_epochs : int, optional
            Number of warmup epoch, by default `0`.
        name : str, optional
            Name of the schedule, by default `None`.
        """
        super().__init__()
        self.scheduler = tf.keras.experimental.CosineDecayRestarts(
            initial_learning_rate=initial_learning_rate,
            first_decay_steps=first_decay_steps,
            t_mul=t_mul,
            m_mul=m_mul,
            alpha=alpha,
            name=name,
        )
        self.initial_learning_rate = initial_learning_rate
        if steps_per_epoch is None:
            raise ValueError('Arg `steps_per_epoch` is None.')
        self.steps_per_epoch = steps_per_epoch
        self.warmup_epochs = warmup_epochs
        self.warmup_steps = int(warmup_epochs * steps_per_epoch)

    def __call__(self, step):
        """Get scheduled learning rate."""
        if self.warmup_steps:
            lr = tf.cond(
                step < self.warmup_steps,
                lambda: self.initial_learning_rate
                * tf.cast(step, tf.float32)
                / tf.cast(self.warmup_steps, tf.float32),
                lambda: self.scheduler(step - self.warmup_steps),
            )
        else:
            lr = self.scheduler(step)

        return lr

    def get_config(self):
        """Get configuration of the schedule."""
        return {
            **self.scheduler.get_config(),
            **{
                'steps_per_epoch': self.steps_per_epoch,
                'warmup_epochs': self.warmup_epochs,
            },
        }
