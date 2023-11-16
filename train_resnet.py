import argparse
import logging
import os, sys
from datetime import datetime

import random
import numpy as np
import torch
from backbones import get_model
from dataset import get_dataloader
from losses import CombinedMarginLoss
from lr_scheduler import PolynomialLRWarmup
from partial_fc_v2 import PartialFC_V2
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.utils_callbacks import CallBackLogging, CallBackEpochLogging, CallBackVerification, EvaluatorLogging
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import AverageMeter, init_logging
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook

# Commented by Bernardo in order to use torch==1.10.1
# assert torch.__version__ >= "1.12.0", "In order to enjoy the features of the new torch, \
# we have upgraded the torch to 1.12.0. torch before than 1.12.0 may not work in the future."


world_size = 1
rank = 0
local_rank = 0


try:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # distributed.init_process_group("nccl")
    distributed.init_process_group("gloo")
except KeyError:
    rank = 0
    local_rank = 0
    world_size = 1
    distributed.init_process_group(
        # backend="nccl",
        backend="gloo",
        # init_method="tcp://127.0.0.1:12584",    # original
        init_method="tcp://127.0.0.1:" + str(int(random.random() * 10000 + 12000)),    # Bernardo
        rank=rank,
        world_size=world_size,
    )


def main(args):

    # get config
    cfg = get_config(args.config)
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    # torch.cuda.set_device(local_rank)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )
    
    wandb_logger = None
    if cfg.using_wandb:
        import wandb
        # Sign in to wandb
        try:
            wandb.login(key=cfg.wandb_key)
        except Exception as e:
            print("WandB Key must be provided in config file (base.py).")
            print(f"Config Error: {e}")
        # Initialize wandb
        run_name = datetime.now().strftime("%y%m%d_%H%M") + f"_GPU{rank}"
        run_name = run_name if cfg.suffix_run_name is None else run_name + f"_{cfg.suffix_run_name}"
        try:
            wandb_logger = wandb.init(
                entity = cfg.wandb_entity, 
                project = cfg.wandb_project, 
                sync_tensorboard = True,
                resume=cfg.wandb_resume,
                name = run_name, 
                notes = cfg.notes) if rank == 0 or cfg.wandb_log_all else None
            if wandb_logger:
                wandb_logger.config.update(cfg)
        except Exception as e:
            print("WandB Data (Entity and Project name) must be provided in config file (base.py).")
            print(f"Config Error: {e}")

    print(f'Loading train paths (dataset: \'{cfg.train_dataset}\')...')
    train_loader = get_dataloader(
        # cfg.rec,          # original
        cfg.train_dataset,  # Bernardo
        cfg.protocol_id,    # Bernardo
        cfg.dataset_path,   # Bernardo
        '',                 # Bernardo
        cfg.img_size,       # Bernardo
        'train',
        local_rank,
        cfg.batch_size,
        cfg.dali,
        cfg.dali_aug,
        cfg.seed,
        cfg.num_workers
    )
    print(f'    train samples: {len(train_loader.dataset)}')

    print(f'Loading val paths (dataset: \'{cfg.train_dataset}\')...')
    val_loader = get_dataloader(
        # cfg.rec,          # original
        cfg.train_dataset,  # Bernardo
        cfg.protocol_id,    # Bernardo
        cfg.dataset_path,   # Bernardo
        '',                 # Bernardo
        cfg.img_size,       # Bernardo
        'val',
        local_rank,
        cfg.batch_size,
        cfg.dali,
        cfg.dali_aug,
        cfg.seed,
        cfg.num_workers
    )
    print(f'    val samples: {len(val_loader.dataset)}')

    print(f'\nBuilding model \'{cfg.network}\'...')
    backbone = get_model(
        cfg.network, img_size=cfg.img_size, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size)

    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=None, bucket_cap_mb=16,
        find_unused_parameters=True)
    backbone.register_comm_hook(None, fp16_compress_hook)

    backbone.train()
    # FIXME using gradient checkpoint if there are some unused parameters will cause error
    backbone._set_static_graph()

    print(f'\nSetting loss function...')
    margin_loss = CombinedMarginLoss(
        64,
        cfg.margin_list[0],
        cfg.margin_list[1],
        cfg.margin_list[2],
        cfg.interclass_filtering_threshold
    )

    print(f'\nSetting optimizer...')
    if cfg.optimizer == "sgd":
        module_partial_fc = PartialFC_V2(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, False)
        # module_partial_fc.train().cuda()
        opt = torch.optim.SGD(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "adamw":
        module_partial_fc = PartialFC_V2(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, False)
        module_partial_fc.train().cuda()
        opt = torch.optim.AdamW(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise

    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.max_epoch

    lr_scheduler = PolynomialLRWarmup(
        optimizer=opt,
        warmup_iters=cfg.warmup_step,
        total_iters=cfg.total_step)

    start_epoch = 0
    global_step = 0
    if cfg.resume:
        dict_checkpoint = torch.load(os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
        module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        opt.load_state_dict(dict_checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        del dict_checkpoint

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    # original
    # callback_verification = CallBackVerification(
    #     val_targets=cfg.val_targets, rec_prefix=cfg.rec,
    #     summary_writer=summary_writer, wandb_logger = wandb_logger
    # )
    # callback_logging = CallBackLogging(
    #     frequent=cfg.frequent,
    #     total_step=cfg.total_step,
    #     batch_size=cfg.batch_size,
    #     start_step = global_step,
    #     writer=summary_writer
    # )

    # Bernardo
    callback_logging = CallBackEpochLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=len(train_loader),
        num_batches=cfg.batch_size,
        start_step = global_step,
        writer=summary_writer
    )

    train_evaluator = EvaluatorLogging(num_samples=len(train_loader.dataset),
                                       batch_size=cfg.batch_size,
                                       num_batches=len(train_loader))
    
    val_evaluator = EvaluatorLogging(num_samples=len(val_loader.dataset),
                                     batch_size=cfg.batch_size,
                                     num_batches=len(val_loader))

    loss_am = AverageMeter()
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    print(f'\nStarting training...')
    for epoch in range(start_epoch, cfg.max_epoch):
        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)
        for train_idx, (img, local_labels) in enumerate(train_loader):             # original
            backbone.train()            # Bernardo
            module_partial_fc.train()   # Bernardo

            global_step += 1
            local_embeddings = backbone(img)
            # loss: torch.Tensor = module_partial_fc(local_embeddings, local_labels)   # original            
            loss, pred_labels = module_partial_fc(local_embeddings, local_labels)      # Bernardo

            if cfg.fp16:
                amp.scale(loss).backward()
                if global_step % cfg.gradient_acc == 0:
                    amp.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    amp.step(opt)
                    amp.update()
                    opt.zero_grad()
            else:
                loss.backward()
                if global_step % cfg.gradient_acc == 0:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    opt.step()
                    opt.zero_grad()

            lr_scheduler.step()
            loss_am.update(loss.item(), 1)

            train_evaluator.update(pred_labels, local_labels)

            print(f'train_idx: {train_idx}')
            with torch.no_grad():
                if wandb_logger:
                    wandb_logger.log({
                        # 'Loss/Step Loss': loss.item(),
                        'Loss/Train Loss': loss_am.avg,
                        # 'Process/Step': global_step,
                        'Process/Epoch': epoch
                    })

                # print('Train:    train_loss:', loss_am.avg)
                callback_logging(global_step, loss_am, train_evaluator, epoch, cfg.fp16, lr_scheduler.get_last_lr()[0], amp)
                loss_am.reset()
                train_evaluator.reset()

                validate(module_partial_fc, backbone, val_loader, val_evaluator, global_step, epoch, summary_writer)   # Bernardo
                print('--------------')


        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.module.state_dict(),
                "state_dict_softmax_fc": module_partial_fc.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))

        if rank == 0:
            path_module = os.path.join(cfg.output, "model.pt")
            torch.save(backbone.module.state_dict(), path_module)

            if wandb_logger and cfg.save_artifacts:
                artifact_name = f"{run_name}_E{epoch}"
                model = wandb.Artifact(artifact_name, type='model')
                model.add_file(path_module)
                wandb_logger.log_artifact(model)
                
        if cfg.dali:
            train_loader.reset()

    if rank == 0:
        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(backbone.module.state_dict(), path_module)
        
        if wandb_logger and cfg.save_artifacts:
            artifact_name = f"{run_name}_Final"
            model = wandb.Artifact(artifact_name, type='model')
            model.add_file(path_module)
            wandb_logger.log_artifact(model)



# Bernardo
def validate(module_partial_fc, backbone, val_loader, val_evaluator, global_step, epoch, writer):
    with torch.no_grad():
        module_partial_fc.eval()
        backbone.eval()
        val_evaluator.reset()

        val_loss_am = AverageMeter()
        for val_idx, (val_img, val_labels) in enumerate(val_loader):
            val_embeddings = backbone(val_img)
            val_loss, pred_labels = module_partial_fc(val_embeddings, val_labels)
            val_loss_am.update(val_loss.item(), 1)
            val_evaluator.update(pred_labels, val_labels)
        
        val_metrics = val_evaluator.evaluate()

        writer.add_scalar('loss/val_loss', val_loss_am.avg, epoch)
        writer.add_scalar('acc/val_acc', val_metrics['acc'], epoch)

        print('Validation:    val_loss: %.4f    val_acc: %.4f%%' % (val_loss_am.avg, val_metrics['acc']))
        val_loss_am.reset()



if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, default='configs/oulu-npu_frames_3d_hrn_r18.py', help="Ex: --config configs/oulu-npu_frames_3d_hrn_r18.py")
    main(parser.parse_args())
