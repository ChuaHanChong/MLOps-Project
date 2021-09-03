"""Module for tfx pipeline."""
import inspect
from typing import Dict
from typing import List
from typing import Optional

import tensorflow_model_analysis as tfma
from icecream import ic
from ml.data import preprocessing_fn
from ml.trainer import run_fn
from ml.utils import LABEL_KEY
from ml.utils import transformed_name
from typeguard import typechecked

from tfx import v1 as tfx
from tfx.orchestration import metadata

# TODO add tfrecords generation
# TODO check log message


@typechecked
def create_pipeline(
    pipeline_name: str,
    pipeline_root: str,
    tfrecords_root: str,
    serving_model_dir: str,
    metadata_path: str,
    preprocessing_fn_custom_config: Dict,
    run_fn_custom_config: Dict,
    beam_pipeline_args: Optional[List[str]],
) -> tfx.dsl.Pipeline:
    """Implement pipeline using TFX."""
    input_config = tfx.proto.Input(
        splits=[
            tfx.proto.Input.Split(name='train', pattern='train/*'),
            tfx.proto.Input.Split(name='eval', pattern='test/*'),
        ],
    )

    # Brings data into the pipeline.
    example_gen = tfx.components.ImportExampleGen(
        input_base=tfrecords_root,
        input_config=input_config,
    )

    # Computes statistics over data for visualization and example validation.
    statistics_gen = tfx.components.StatisticsGen(
        examples=example_gen.outputs['examples'],
    )

    # Generates schema based on statistics files.
    schema_gen = tfx.components.SchemaGen(
        statistics=statistics_gen.outputs['statistics'],
        infer_feature_shape=True,
    )

    # Performs anomaly detection based on statistics and data schema.
    example_validator = tfx.components.ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema'],
    )

    # Performs transformations and feature engineering in training and serving.
    transform = tfx.components.Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=inspect.getfile(preprocessing_fn),
        custom_config=preprocessing_fn_custom_config,
    )

    # Uses user-provided Python function that trains a model using TF-Learn.
    trainer = tfx.components.Trainer(
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        module_file=inspect.getfile(run_fn),
        train_args=tfx.proto.TrainArgs(
            num_steps=run_fn_custom_config['steps_per_epoch']
            * run_fn_custom_config['num_epochs'],
        ),
        eval_args=tfx.proto.EvalArgs(
            num_steps=run_fn_custom_config['validation_steps'],
        ),
        custom_config=run_fn_custom_config,
    )

    # Get the latest blessed model for model validation.
    model_resolver = tfx.dsl.Resolver(
        strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
        model=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Model),
        model_blessing=tfx.dsl.Channel(type=tfx.types.standard_artifacts.ModelBlessing),
    ).with_id('latest_blessed_model_resolver')

    # Uses TFMA to compute evaluation statistics over features of a model and
    # perform quality validation of a candidate model (compare to a baseline).
    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key=transformed_name(LABEL_KEY))],
        slicing_specs=[tfma.SlicingSpec()],
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[
                    tfma.MetricConfig(
                        class_name='SparseCategoricalAccuracy',
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={'value': 0.001},
                            ),
                            # Change threshold will be ignored if there is no
                            # baseline model resolved from MLMD (first run).
                            change_threshold=tfma.GenericChangeThreshold(
                                direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                absolute={'value': -1e-3},
                            ),
                        ),
                    ),
                ],
            ),
        ],
    )
    evaluator = tfx.components.Evaluator(
        examples=transform.outputs['transformed_examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config,
    )

    # Checks whether the model passed the validation steps and pushes the model
    # to a file destination if check passed.
    pusher = tfx.components.Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=tfx.proto.PushDestination(
            filesystem=tfx.proto.PushDestination.Filesystem(
                base_directory=serving_model_dir,
            ),
        ),
    )

    components = [
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        trainer,
        model_resolver,
        evaluator,
        pusher,
    ]

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path,
        ),
        beam_pipeline_args=beam_pipeline_args,
    )
