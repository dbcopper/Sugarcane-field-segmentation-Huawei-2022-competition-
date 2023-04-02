import os
import platform

enable_auto_install = True

def _get_rank() -> int:
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


if enable_auto_install and 0 == _get_rank():
    os.system("pip install torchvision==0.12.0 --upgrade")
    os.system("pip install matplotlib")
    os.system("pip install pytorch-lightning")
    os.system("pip install torchmetrics")
    os.system("pip install opencv-python")

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))


from model_wrapper import UNetWrapper
from dataset import RemoteSenseDataset
from torchvision import transforms
import torch.utils.data as torch_data
import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from callbacks.image_show_in_tfboard import TensorboardOnRemoteSense
import torch.nn.functional as F
import utils
from functools import partial
from config import Config
import torch
import cv2


def setup_multi_processes():
    """Setup multi-processing environment variables."""
    # disable opencv multithreading to avoid system being overloaded
    cv2.setNumThreads(0)


def bgr2rgb(x):
        return x[..., ::-1].copy()

def main():
    max_epoch = Config.get('max_epoch', 50)
    model = UNetWrapper({
        'lr': Config.get('lr', 7e-4)
    })

    dataset_mean = [0.19081172, 0.19135737, 0.2075335]
    dataset_std = [0.07755211, 0.06542103, 0.07392063]
    block_size = 512
    
    data_url = Config.get('data_url', './dataset')
    print(os.listdir(data_url))
    train_dataset_path = os.path.join(data_url, 'train')
    test_dataset_path = os.path.join(data_url, 'split_imgs')
    
    train_dataset = RemoteSenseDataset(train_dataset_path, block_size = block_size, enable_cache=True, transform=transforms.Compose([
        bgr2rgb,
        transforms.ToTensor(),
        transforms.Normalize(mean = dataset_mean, std = dataset_std)
    ]))

    # 验证集数据
    test_dataset = RemoteSenseDataset(test_dataset_path, block_size = block_size, test_mode=True, enable_cache=True, transform=transforms.Compose([
        bgr2rgb,
        transforms.ToTensor(),
        transforms.Normalize(mean = dataset_mean, std = dataset_std)
    ]))
    
    cpu_count = os.cpu_count()
    train_batch_size = 2
    test_batch_size = 1
    train_num_workers = min(train_batch_size // 4, cpu_count)
    test_num_workers = min(test_batch_size, cpu_count)

    
    train_loader = torch_data.DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size, num_workers=train_num_workers)
    val_loader = torch_data.DataLoader(test_dataset, shuffle=False, batch_size=test_batch_size, num_workers=test_num_workers)

    img_show_tb_callback = TensorboardOnRemoteSense(
        postprocess_hook=partial(torch.argmax, dim=1),
        transform_hook=utils.Compose([
            utils.DeNormalizeAndShow(num_classes=2, mean=dataset_mean, std=dataset_std),
            utils.BGR2RGB(dim=1),
        ])
    )

    model_save_path = Config.get('TRAIN_URL', './log')
    model_ckpt_path = os.path.join(model_save_path, 'checkpoints')
    os.makedirs(model_ckpt_path, exist_ok = True)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_miou_epoch',
        dirpath=model_ckpt_path,
        filename='unet-{epoch:02d}-{val_miou_epoch:.3f}',
        save_top_k=3,
        mode='max',
    )

    gpu_number = torch.cuda.device_count()
    if gpu_number > 1:
        sync_batchnorm = True
        arg_dicts = dict(strategy='ddp', sync_batchnorm=sync_batchnorm)
    else:
        arg_dicts = dict()

    logger = TensorBoardLogger(os.path.join(model_save_path, "tb_logs"), name="rs_unet")
    trainer = pl.Trainer(gpus=gpu_number, max_epochs=max_epoch,
                         val_check_interval=1.0, logger=logger, 
                         callbacks = [checkpoint_callback, img_show_tb_callback], **arg_dicts
                        )
                        
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()