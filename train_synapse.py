import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from configs.config_synapse import config
from datasets import SynapseDataset, SynapseRandomGenerator
from engine import train_one_epoch, validate
from losses import ACDRMUNetLoss
from models.acdrmunet import ACDRMUNet, load_vmamba_small_pretrained


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_logger(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("acdrmunet")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def build_loaders():
    train_dataset = SynapseDataset(
        base_dir=config.data_path,
        list_dir=config.list_dir,
        split="train",
        transform=SynapseRandomGenerator((config.input_size, config.input_size)),
    )
    validation_dataset = SynapseDataset(
        base_dir=config.volume_path,
        list_dir=config.list_dir,
        split="test_vol",
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=config.num_workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )
    return train_loader, validation_loader


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_dice):
    torch.save(
        {
            "epoch": epoch,
            "best_dice": best_dice,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        path,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train ACDRMUNet on Synapse")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    set_seed(config.seed)

    if args.resume:
        run_dir = args.resume.expanduser().resolve().parent
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = config.output_dir / f"synapse_{timestamp}"
    logger = create_logger(run_dir)
    train_loader, validation_loader = build_loaders()

    model = ACDRMUNet(**config.model).to(device)
    criterion = ACDRMUNetLoss(auxiliary_weight=0.2).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.scheduler_t_max,
        eta_min=config.scheduler_eta_min,
    )

    start_epoch = 1
    best_dice = -1.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_dice = float(checkpoint.get("best_dice", -1.0))
        logger.info("Resumed from %s", args.resume)
    elif not args.no_pretrained:
        report = load_vmamba_small_pretrained(model, config.pretrained_path)
        logger.info("Pretrained: %s", json.dumps(report))

    logger.info("Run directory: %s", run_dir)
    logger.info("Train samples: %d", len(train_loader.dataset))
    logger.info("Validation cases: %d", len(validation_loader.dataset))

    for epoch in range(start_epoch, config.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        scheduler.step()
        logger.info(
            "Epoch %d/%d | loss %.6f | lr %.8f",
            epoch,
            config.epochs,
            train_loss,
            optimizer.param_groups[0]["lr"],
        )

        if epoch % config.validation_interval == 0 or epoch == config.epochs:
            metrics = validate(
                model,
                validation_loader,
                device,
                config.input_size,
                config.num_classes,
            )
            logger.info(
                "Validation | epoch %d | Dice %.6f | HD95 %.6f",
                epoch,
                metrics["dice"],
                metrics["hd95"],
            )
            for name, values in metrics["per_class"].items():
                logger.info(
                    "%s | Dice %.6f | HD95 %.6f",
                    name,
                    values["dice"],
                    values["hd95"],
                )
            if metrics["dice"] > best_dice:
                best_dice = metrics["dice"]
                torch.save(model.state_dict(), run_dir / "best.pth")

        save_checkpoint(
            run_dir / "latest.pth",
            model,
            optimizer,
            scheduler,
            epoch,
            best_dice,
        )


if __name__ == "__main__":
    main()
