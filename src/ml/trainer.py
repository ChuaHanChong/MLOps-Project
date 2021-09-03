"""Module for model trainer."""
import re
from pathlib import Path
from typing import List
from typing import Optional

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_model_optimization as tfmot
import tensorflow_transform as tft
from ml.image import AugmentImage
from ml.model import build_model
from ml.model import default_trainable_expr
from ml.training.callbacks import AverageModelCheckpoint
from ml.training.callbacks import EarlyStopping
from ml.training.callbacks import LearningRateLogger
from ml.training.callbacks import ReduceLROnPlateau
from ml.training.lr_schedules import WarmupCosineDecayRestarts
from ml.training.prunning import apply_pruning_to_layer
from ml.training.utils import call_loss_object
from ml.training.utils import call_metric_object
from ml.utils import IMAGE_KEY
from ml.utils import LABEL_KEY
from ml.utils import transformed_name
from tensorflow.python.keras.utils.layer_utils import count_params
from tfx_bsl.public import tfxio

from tfx import v1 as tfx


# default setting
DEFAULT_LR_CONFIG = {
    'lr_base': 0.016,  # base learning rate
    'use_lr_scheduler': True,  # use cosine decay lr with restarts & warmup
    'lr_warmup_epoch': 5,  # number of learning rate warmup epoch
    'epochs_per_restart': 25,  # number of epoch to restart cosine scheduler
}
DEFAULT_MONITOR_CONFIG = {
    'metric': 'val_loss',  # monitoring metric for keras callbacks
    'mode': 'min',  # monitoring mode for keras callbacks
    'baseline': None,  # baseline value for the monitored metric
    'target': None,  # target value for the monitored metric
    'min_delta': 0.0001,  # minimum improvement for monitoring metric
}


class TrainWorker:
    """Model training worker."""

    def __init__(
        self,
        fn_args: tfx.components.FnArgs,
        name: Optional[str] = None,
    ):
        """Initialize.

        Parameters
        ----------
        config : dict
            Model training configuration.
        name : str, optional
            Name of the training model, by default `None`.
        """
        self.fn_args = fn_args
        self.name = name

        self._setup_strategy()

    def _setup_strategy(self):
        """Configure distributed strategy."""
        if tf.config.list_physical_devices('GPU'):
            gpus = tf.config.list_physical_devices('GPU')
            num_gpus = len(gpus)
            if num_gpus > 1:
                self.fn_args.custom_config['batch_size'] *= num_gpus
                ds_strategy = tf.distribute.MirroredStrategy()
            else:
                ds_strategy = tf.distribute.OneDeviceStrategy('device:GPU:0')
        else:
            ds_strategy = tf.distribute.OneDeviceStrategy('device:CPU:0')
        self.ds_strategy = ds_strategy

    def _setup_input_fn(
        self,
        file_pattern: List[str],
        data_accessor: tfx.components.DataAccessor,
        tf_transform_output: tft.TFTransformOutput,
        is_train: bool = False,
        batch_size: int = 256,
    ):
        """Generate features and label for tuning/training."""
        dataset = data_accessor.tf_dataset_factory(
            file_pattern,
            tfxio.TensorFlowDatasetOptions(
                batch_size=batch_size,
                label_key=transformed_name(LABEL_KEY),
            ),
            tf_transform_output.transformed_metadata.schema,
        )
        # Apply data augmentation. We have to do data augmentation here
        # because we need to apply data agumentation on-the-fly during training.
        # If we put it in Transform, it will only be applied once on the whole dataset,
        # which will lose the point of data augmentation.
        augment_image = AugmentImage(
            augname='randaug',
            ra_num_layers=1,
            ra_magnitude=15,
        )

        def _augment_data(feature_dict):
            image_features = feature_dict[transformed_name(IMAGE_KEY)]
            image_features = augment_image(image_features)
            feature_dict[transformed_name(IMAGE_KEY)] = image_features
            return feature_dict

        if is_train:
            dataset = dataset.map(lambda x, y: (_augment_data(x), y))

        return dataset

    def _setup_callbacks(
        self,
        run_dir,
        steps_per_epoch,
        prune_model=False,
        verbose=1,
    ):
        """Configure training callbacks."""
        lr_logger = LearningRateLogger(steps_per_epoch)
        if prune_model:
            pruning_callbacks = [
                tfmot.sparsity.keras.UpdatePruningStep(),
                tfmot.sparsity.keras.PruningSummaries(
                    log_dir=str(run_dir.joinpath('logs')),
                ),
            ]
            callbacks = [lr_logger, *pruning_callbacks]
        else:
            tb_callback = tf.keras.callbacks.TensorBoard(
                log_dir=str(run_dir.joinpath('logs')),
            )
            callbacks = [lr_logger, tb_callback]

        avg_ckpt_callback = AverageModelCheckpoint(
            filepath=str(run_dir.joinpath('checkpoint', 'emackpt-{epoch:d}')),
            monitor=self.fn_args.custom_config['monitor'].get(
                'metric',
                DEFAULT_MONITOR_CONFIG['metric'],
            ),
            verbose=verbose,
            save_best_only=True,
            save_weights_only=True,
            update_weights=True,
            mode=self.fn_args.custom_config['monitor'].get(
                'mode',
                DEFAULT_MONITOR_CONFIG['mode'],
            ),
        )
        es_callback = EarlyStopping(
            monitor=self.fn_args.custom_config['monitor'].get(
                'metric',
                DEFAULT_MONITOR_CONFIG['metric'],
            ),
            min_delta=self.fn_args.custom_config['monitor'].get(
                'min_delta',
                DEFAULT_MONITOR_CONFIG['min_delta'],
            ),
            patience=self.fn_args.custom_config['learning_rate'].get(
                'epochs_per_restart',
                DEFAULT_LR_CONFIG['epochs_per_restart'],
            ),
            verbose=verbose,
            mode=self.fn_args.custom_config['monitor'].get(
                'mode',
                DEFAULT_MONITOR_CONFIG['mode'],
            ),
            target=self.fn_args.custom_config['monitor'].get(
                'target',
                DEFAULT_MONITOR_CONFIG['target'],
            ),
            restore_best_weights=True,
        )
        rlrop_factor = 0.8
        rlrop_callback = ReduceLROnPlateau(
            monitor=self.fn_args.custom_config['monitor'].get(
                'metric',
                DEFAULT_MONITOR_CONFIG['metric'],
            ),
            factor=rlrop_factor,
            patience=int(
                self.fn_args.custom_config['learning_rate'].get(
                    'epochs_per_restart',
                    DEFAULT_LR_CONFIG['epochs_per_restart'],
                )
                * rlrop_factor,
            ),
            verbose=verbose,
            mode=self.fn_args.custom_config['monitor'].get(
                'mode',
                DEFAULT_MONITOR_CONFIG['mode'],
            ),
            min_delta=self.fn_args.custom_config['monitor'].get(
                'min_delta',
                DEFAULT_MONITOR_CONFIG['min_delta'],
            ),
            steps_per_epoch=steps_per_epoch,
        )

        callbacks.append([avg_ckpt_callback, rlrop_callback, es_callback])
        return callbacks

    def _setup_optimizer(self, steps_per_epoch, lr=None):
        """Configure default learning rate schedule and optimizer."""
        batch_size = self.fn_args.custom_config['batch_size']
        batch_size_per_replica = batch_size // self.ds_strategy.num_replicas_in_sync

        scaled_lr = (lr or DEFAULT_LR_CONFIG['lr_base']) * (
            batch_size_per_replica / 256.0
        )
        if self.fn_args.custom_config['learning_rate'].get(
            'use_lr_scheduler',
            DEFAULT_LR_CONFIG['use_lr_scheduler'],
        ):
            learning_rate = WarmupCosineDecayRestarts(
                scaled_lr,
                first_decay_steps=steps_per_epoch
                * self.fn_args.custom_config['learning_rate'].get(
                    'epochs_per_restart',
                    DEFAULT_LR_CONFIG['epochs_per_restart'],
                ),
                t_mul=1.0,
                m_mul=1.0,
                steps_per_epoch=steps_per_epoch,
                warmup_epochs=self.fn_args.custom_config['learning_rate'].get(
                    'lr_warmup_epoch',
                    DEFAULT_LR_CONFIG['lr_warmup_epoch'],
                ),
            )
        else:
            learning_rate = scaled_lr

        optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=0.9,
            nesterov=True,
            clipnorm=1.0,
        )
        optimizer = tfa.optimizers.MovingAverage(optimizer, dynamic_decay=True)

        return optimizer

    def _setup_model(
        self,
        model,
        loss,
        metrics,
        steps_per_epoch,
        lr=None,
        var_trainable_expr=None,
        ckpt_dir=None,
        prune_model=False,
        pruned_ckpt_dir=None,
        verbose=1,
    ):
        """Configure model for training.

        Configure model weight and trainable layers, prune model, and compile model.
        """
        optimizer = self._setup_optimizer(steps_per_epoch, lr=lr)

        if var_trainable_expr:
            for layer in model.layers:
                if re.match(var_trainable_expr, layer.name):
                    if not isinstance(layer, tf.keras.layers.BatchNormalization):
                        layer.trainable = True

        if ckpt_dir:
            ckpt_path = tf.train.latest_checkpoint(ckpt_dir)
            print(f'Load checkpoint: {ckpt_path}')
            model.load_weights(ckpt_path)

        if prune_model:
            model = tf.keras.models.clone_model(
                model,
                clone_function=apply_pruning_to_layer,
            )
            if pruned_ckpt_dir:
                pruned_ckpt_path = tf.train.latest_checkpoint(pruned_ckpt_dir)
                print(f'Load checkpoint: {pruned_ckpt_path}')
                model.load_weights(pruned_ckpt_path)

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        if verbose > 1:
            print(model.summary())
        elif verbose > 0:
            total_params = count_params(model.weights)
            trainable_params = count_params(model.trainable_weights)
            xtrainable_params = count_params(model.non_trainable_weights)
            print('=' * 50)
            print(f'Total params: {total_params:,}')
            print(f'Trainable params: {trainable_params:,}')
            print(f'Non-trainable params: {xtrainable_params:,}')
            print('=' * 50)

        return model

    def _save_model(
        self,
        model,
        model_dir,
        prune_model=False,
    ):
        """Save model in TF SavedModel format."""
        # TF SavedModel format
        if prune_model:
            model_for_export = tfmot.sparsity.keras.strip_pruning(model)
        else:
            model_for_export = model

        model_for_export.save(model_dir, include_optimizer=False, save_format='tf')

    def _fit(
        self,
        model,
        loss,
        metrics,
        train_dataset,
        val_dataset,
        run_dir,
        var_trainable_expr=None,
        ckpt_dir=None,
        prune_model=False,
        pruned_ckpt_dir=None,
    ):
        """Train model."""
        num_epochs = self.fn_args.custom_config['num_epochs']
        steps_per_epoch = self.fn_args.custom_config['steps_per_epoch']
        validation_steps = self.fn_args.custom_config['validation_steps']

        model = self._setup_model(
            model,
            loss,
            metrics,
            steps_per_epoch,
            lr=self.fn_args.custom_config['learning_rate'].get('base_lr', None),
            var_trainable_expr=var_trainable_expr,
            ckpt_dir=ckpt_dir,
            prune_model=prune_model,
            pruned_ckpt_dir=pruned_ckpt_dir,
        )
        callbacks = self._setup_callbacks(
            run_dir,
            steps_per_epoch,
            prune_model=prune_model,
        )
        model.fit(
            train_dataset,
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks,
            validation_data=val_dataset,
            validation_steps=validation_steps,
        )
        best_ckpt_path = tf.train.latest_checkpoint(str(run_dir / 'checkpoint'))
        model.load_weights(best_ckpt_path)

        return model

    def train(self, save_model=True):
        """Train model with two steps training."""
        tf_transform_output = tft.TFTransformOutput(
            self.fn_args.transform_output,
        )

        train_dataset = self._setup_input_fn(
            self.fn_args.train_files,
            self.fn_args.data_accessor,
            tf_transform_output,
            is_train=True,
            batch_size=self.fn_args.custom_config['batch_size'],
        )
        val_dataset = self._setup_input_fn(
            self.fn_args.eval_files,
            self.fn_args.data_accessor,
            tf_transform_output,
            is_train=False,
            batch_size=self.fn_args.custom_config['batch_size'],
        )

        with self.ds_strategy.scope():
            model = build_model(
                self.fn_args.custom_config['model_name'],
                self.fn_args.custom_config['num_classes'],
                pretrained_weight=self.fn_args.custom_config.get(
                    'pretrained_path',
                    None,
                ),
                output_activation=self.fn_args.custom_config.get(
                    'output_activation',
                    None,
                ),
                name=self.name,
            )
            if self.fn_args.custom_config.get('pretrained_path', None):
                pretrained_weight = self.fn_args.custom_config['pretrained_path']
                print(f'Load pretrained weight: {pretrained_weight}')

            loss = call_loss_object(self.fn_args.custom_config['loss'])
            metrics = [
                call_metric_object(metric_config)
                for metric_config in self.fn_args.custom_config['metrics']
            ]

            if self.fn_args.custom_config['transfer_learning']:
                # Only train newly added head layers
                print(f'{self.name} - transferlearning...')
                model = self._fit(
                    model,
                    loss,
                    metrics,
                    train_dataset,
                    val_dataset,
                    Path(self.fn_args.model_run_dir) / 'transferred',
                    ckpt_dir=self.fn_args.custom_config.get('ckpt_dir', None),
                )

            if self.fn_args.custom_config['fine_tuning']:
                # Finetune model & prune model
                print(f'{self.name} - finetuning...')
                model = self._fit(
                    model,
                    loss,
                    metrics,
                    train_dataset,
                    val_dataset,
                    Path(self.fn_args.model_run_dir) / 'finetuned',
                    var_trainable_expr=self.fn_args.custom_config.get(
                        'var_trainable_expr',
                        None,
                    )
                    or default_trainable_expr[self.fn_args.custom_config['model_name']],
                    prune_model=True,
                    pruned_ckpt_dir=self.fn_args.custom_config.get(
                        'pruned_ckpt_dir',
                        None,
                    ),
                )

            if save_model:
                self._save_model(
                    model,
                    self.fn_args.serving_model_dir,
                    prune_model=self.fn_args.custom_config['fine_tuning'],
                )


# TFX Trainer will call this function.
def run_fn(fn_args: tfx.components.FnArgs):
    """
    Train the model based on given args.

    Parameters
    ----------
    fn_args : tfx.components.FnArgs
        Holds args used to train the model as name/value pairs.
    """
    # set default setting for learning rate and metric monitoring
    if not fn_args.custom_config.get('learning_rate', None):
        fn_args.custom_config['learning_rate'] = DEFAULT_LR_CONFIG
    if not fn_args.custom_config.get('monitor', None):
        fn_args.custom_config['monitor'] = DEFAULT_MONITOR_CONFIG

    worker = TrainWorker(
        fn_args,
        name='Model{name}'.format(
            name=fn_args.custom_config['name'].capitalize(),
        ),
    )
    worker.train()
