"""Train a classification model."""
import argparse
from pathlib import Path

import yaml
from icecream import ic
from ml.model import input_shapes
from ml.pipeline import create_pipeline
from tfx import v1 as tfx


def main(args):
    """Execute main function."""
    with open(args.config) as stream:
        config = yaml.safe_load(stream)
        ic(config)

    pipeline_name = config['name']

    root = Path.cwd().joinpath('data')
    data_root = root.joinpath('raw', pipeline_name)

    tfx_root = Path.cwd().joinpath('data', 'tfx')
    pipeline_root = tfx_root.joinpath('pipelines', pipeline_name)
    metadata_path = tfx_root.joinpath('metadata', pipeline_name, 'metadata.db')

    # Pipeline arguments for Beam powered Components.
    # beam_pipeline_args = [
    #    '--direct_running_mode=multi_processing',
    #    # 0 means auto-detect based on on the number of CPUs available
    #    # during execution time.
    #    '--direct_num_workers=0',
    # ]
    beam_pipeline_args = None

    preprocess_config = {
        'target_size': input_shapes[config['train']['model_name']],
    }

    num_data = {'train': 128, 'test': 128}
    train_config = config['train']
    train_config['name'] = config['name']
    train_config['steps_per_epoch'] = num_data['train'] // train_config['batch_size']
    train_config['validation_steps'] = num_data['test'] // train_config['batch_size']

    tfx.orchestration.LocalDagRunner().run(
        create_pipeline(
            pipeline_name=pipeline_name,
            pipeline_root=str(pipeline_root),
            data_root=str(data_root),
            metadata_path=str(metadata_path),
            preprocessing_fn_custom_config=preprocess_config,
            run_fn_custom_config=train_config,
            beam_pipeline_args=beam_pipeline_args,
        ),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        dest='config',
        help='Yaml configuration file.',
    )
    args = parser.parse_args()

    main(args)
