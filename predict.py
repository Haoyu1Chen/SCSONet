import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from models.scsonet import SCOUNet
from dataset.npy_datasets import NPY_datasets
from engine import *
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "0, 1, 2, 3"

from utils import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")



def predict(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    set_seed(config.seed)
    gpu_ids = [0]# [0, 1, 2, 3]
    torch.cuda.empty_cache()
    criterion = config.criterion
    log_dir = os.path.join(config.work_dir, 'log')

    global logger
    logger = get_logger('train', log_dir)

    log_config_info(config, logger)



    model_cfg = config.model_config

    model = SCOUNet(num_classes=model_cfg['num_classes'],
                    input_channels=model_cfg['input_channels'],
                    c_list=model_cfg['c_list'],
                    split_att=model_cfg['split_att'],
                    bridge=model_cfg['bridge'])
    model = torch.nn.DataParallel(model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])
    val_dataset = NPY_datasets(config.data_path, config, train=False)

    val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=config.num_workers,
                                drop_last=True)



    best_weight = torch.load('', map_location=torch.device('cpu'))
    model.module.load_state_dict(best_weight)
    loss = test_one_epoch(
        val_loader,
        model,
        criterion,
        logger,
        config,
    )


#     checkpoint_dir = 'results/18 bset 6 2/checkpoints'
#
# # List all files in the checkpoint directory
#     checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
#
#     for checkpoint_file in checkpoint_files:
#         # Load the checkpoint
#         checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
#         best_weight = torch.load(checkpoint_path, map_location=torch.device('cpu'))
#         model.module.load_state_dict(best_weight)
#         loss = test_one_epoch(
#             val_loader,
#             model,
#             criterion,
#             logger,
#             config,
#         )

if __name__ == '__main__':
    config = setting_config  # Load your configuration
    predict(config)
