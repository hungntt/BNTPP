from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from supervisor import Supervisor
import yaml
from pathlib import Path
import os
import json
from models.lib.utils import SetSeed
import matplotlib.pyplot as plt


def main(args):
    config_filename = f'./experiments/{args.dataset}/config.yaml'
    with open(config_filename) as f:
        supervisor_config = yaml.safe_load(f)
        path = Path(supervisor_config['train']['log_dir']) / supervisor_config['train']['experiment_name']
        path.mkdir(exist_ok=True, parents=True)

        sv_param = os.path.join(path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(supervisor_config, file_obj)
            print(supervisor_config)
        seed = supervisor_config['seed']
        SetSeed(seed)

        supervisor = Supervisor(**supervisor_config)

        supervisor.train()
        # Visualize the loss
        plt.plot(supervisor.loss_list_training, label='Training loss')
        plt.plot(supervisor.loss_list_validation, label='Validation loss')
        plt.legend()
        plt.title(
                f'Loss of {supervisor_config["model"]["intensity_type"]}-{supervisor_config["model"]["encoder_type"]}')
        plt.xlabel('Epoch')
        plt.ylabel('NLL Loss')
        plt.show()
        supervisor._test_final_n_epoch(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='helpdesk', type=str,
                        help='Dataset name (default: helpdesk, other options depend on the defined dataset folder name')

    args = parser.parse_args()
    main(args)
