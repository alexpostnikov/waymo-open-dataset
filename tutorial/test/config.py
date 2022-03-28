import argparse


def build_parser():
    """Build parser."""
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)

    ###########################################################################
    # #### Wandb ##############################################################
    ###########################################################################
    parser.add_argument(
        '--wandb_project', type=str, default='waymo22',
        help='Wandb project name.')

    ###########################################################################
    # #### Directories ########################################################
    ###########################################################################
    parser.add_argument(
        '--dir_wandb', type=str, default='./',
        help='Directory to which wandb logs.')

    parser.add_argument(
        '--dir_data', type=str, default='/media/robot/hdd/waymo_dataset/tf_example',
        help='Directory where SDC data is stored. We also use this to cache '
             'torch hub models.')

    parser.add_argument(
        '--dir_checkpoint', type=str, default="./checkpoints/",
        help='Directory to which model checkpoints are stored.')

    ###########################################################################
    # #### General ############################################################
    ###########################################################################

    parser.add_argument(
        '--np_seed', type=int, default=0,
        help='NumPy seed (data processing, train/test splits, etc.).')

    parser.add_argument(
        '--torch_seed', type=int, default=0,
        help='Model seed.')

    ###########################################################################
    # #### Data ###############################################################
    ###########################################################################
    parser.add_argument(
        '--data_num_workers', type=int, default=4,
        help='Number of workers to use in PyTorch data loading.')


    ###########################################################################
    # #### Experiment #########################################################
    ###########################################################################

    parser.add_argument(
        '--exp_name', type=str, default=None,
        help=f'Specify an explicit name for the experiment - otherwise we '
             f'generate a unique wandb ID.')

    parser.add_argument(
        '--use_sgd', type="bool", default=False)
    parser.add_argument(
        '--exp_lr', type=float, default=3e-4)
    parser.add_argument(
        '--exp_num_lr_warmup_steps', type=int, default=1000)
    parser.add_argument(
        '--exp_batch_size', type=int, default=4)
    parser.add_argument(
        '--epoch_to_load', type=int, default=-1)
    parser.add_argument(
        '--exp_num_epochs', type=int, default=5)


    ###########################################################################
    # #### Model ##############################################################
    ###########################################################################



    ###########################################################################
    # #### Method-Specific Hypers #############################################
    ###########################################################################

    ###########################################################################
    # #### Metrics ############################################################
    ###########################################################################


    ###########################################################################
    # #### Debug ##############################################################
    ###########################################################################

    return parser


def str2bool(v):
    """https://stackoverflow.com/questions/15008758/
    parsing-boolean-values-with-argparse/36031646"""
    # susendberg's function
    return v.lower() in ("yes", "true", "t", "1")
