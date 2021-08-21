"""Module for tfx pipeline."""
import inspect
from typing import Dict
from typing import List
from typing import Optional

from ml.data import preprocessing_fn
from ml.trainer import run_fn
from tfx import v1 as tfx
from tfx.orchestration.metadata import sqlite_metadata_connection_config
from typeguard import typechecked


@typechecked
def create_pipeline(
    pipeline_name: str,
    pipeline_root: str,
    data_root: str,
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
        input_base=data_root,
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

    components = [
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        trainer,
        # model_resolver,
        # evaluator,
        # pusher
    ]

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        metadata_connection_config=sqlite_metadata_connection_config(
            metadata_path,
        ),
        beam_pipeline_args=beam_pipeline_args,
    )
