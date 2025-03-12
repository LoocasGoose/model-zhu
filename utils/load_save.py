import os
import shutil

import torch


def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    """
    Load checkpoint and ,if possible, schedulers and optimizer.
    """
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(config.MODEL.RESUME, map_location="cpu", check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location="cpu")
    msg = model.load_state_dict(checkpoint["model"], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and "optimizer" in checkpoint and "lr_scheduler" in checkpoint and "epoch" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint["epoch"] + 1
        config.freeze()
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if "max_accuracy" in checkpoint:
            max_accuracy = checkpoint["max_accuracy"]

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger, is_best=False):
    """
    Saves checkpoint of the model.
    
    Args:
        config: Configuration object
        epoch: Current epoch
        model: Model to save
        max_accuracy: Best accuracy achieved so far
        optimizer: Optimizer state
        lr_scheduler: Learning rate scheduler state
        logger: Logger for output
        is_best: Whether this checkpoint is the best so far
    """
    save_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "max_accuracy": max_accuracy,
        "epoch": epoch,
        "config": config,
    }

    if is_best:
        save_path = os.path.join(config.OUTPUT, "ckpt_best.pth")
        logger.info(f"{save_path} saving (best model)......")
        torch.save(save_state, save_path)
        logger.info(f"{save_path} saved !!!")
    else:
        save_path = os.path.join(config.OUTPUT, f"ckpt_epoch_{epoch}.pth")
        logger.info(f"{save_path} saving......")
        torch.save(save_state, save_path)
        logger.info(f"{save_path} saved !!!")
