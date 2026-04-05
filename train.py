"""Training entrypoint"""

import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import wandb

from data.pets_dataset import OxfordIIITPetDataset
from models import VGG11Classifier, VGG11Localizer, VGG11UNet, MultiTaskPerceptionModel
from losses import IoULoss


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_dataloaders(data_root, image_size, batch_size):
    full_dataset = OxfordIIITPetDataset(
        root=data_root,
        split="trainval",
        image_size=image_size
    )

    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader


def build_model(task, num_classes, seg_classes, dropout_p):
    if task == "classification":
        return VGG11Classifier(num_classes=num_classes, dropout_p=dropout_p)
    elif task == "localization":
        return VGG11Localizer(dropout_p=dropout_p)
    elif task == "segmentation":
        return VGG11UNet(num_classes=seg_classes, dropout_p=dropout_p)
    elif task == "multitask":
        return MultiTaskPerceptionModel(
            num_breeds=num_classes,
            seg_classes=seg_classes,
            dropout_p=dropout_p
        )
    else:
        raise ValueError(f"Unknown task: {task}")


def train_one_epoch_classification(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate_classification(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def train_one_epoch_localization(model, loader, optimizer, mse_loss, iou_loss, device):
    model.train()
    total_loss = 0.0
    total = 0

    for images, _, bboxes, _ in loader:
        images = images.to(device)
        bboxes = bboxes.to(device)

        optimizer.zero_grad()
        pred_boxes = model(images)
        loss = mse_loss(pred_boxes, bboxes) + iou_loss(pred_boxes, bboxes)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total += images.size(0)

    return total_loss / total


@torch.no_grad()
def validate_localization(model, loader, mse_loss, iou_loss, device):
    model.eval()
    total_loss = 0.0
    total = 0

    for images, _, bboxes, _ in loader:
        images = images.to(device)
        bboxes = bboxes.to(device)

        pred_boxes = model(images)
        loss = mse_loss(pred_boxes, bboxes) + iou_loss(pred_boxes, bboxes)

        total_loss += loss.item() * images.size(0)
        total += images.size(0)

    return total_loss / total


def train_one_epoch_segmentation(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total = 0

    for images, _, _, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total += images.size(0)

    return total_loss / total


@torch.no_grad()
def validate_segmentation(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0

    for images, _, _, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        total_loss += loss.item() * images.size(0)
        total += images.size(0)

    return total_loss / total


def train_one_epoch_multitask(model, loader, optimizer, cls_loss_fn, mse_loss, iou_loss, seg_loss_fn, device):
    model.train()
    total_loss = 0.0
    total = 0

    for images, labels, bboxes, masks in loader:
        images = images.to(device)
        labels = labels.to(device)
        bboxes = bboxes.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        cls_loss = cls_loss_fn(outputs["classification"], labels)
        box_loss = mse_loss(outputs["localization"], bboxes) + iou_loss(outputs["localization"], bboxes)
        seg_loss = seg_loss_fn(outputs["segmentation"], masks)

        loss = cls_loss + box_loss + seg_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total += images.size(0)

    return total_loss / total


@torch.no_grad()
def validate_multitask(model, loader, cls_loss_fn, mse_loss, iou_loss, seg_loss_fn, device):
    model.eval()
    total_loss = 0.0
    total = 0

    for images, labels, bboxes, masks in loader:
        images = images.to(device)
        labels = labels.to(device)
        bboxes = bboxes.to(device)
        masks = masks.to(device)

        outputs = model(images)

        cls_loss = cls_loss_fn(outputs["classification"], labels)
        box_loss = mse_loss(outputs["localization"], bboxes) + iou_loss(outputs["localization"], bboxes)
        seg_loss = seg_loss_fn(outputs["segmentation"], masks)

        loss = cls_loss + box_loss + seg_loss

        total_loss += loss.item() * images.size(0)
        total += images.size(0)

    return total_loss / total


def main():
    parser = argparse.ArgumentParser(description="Train visual perception models")
    parser.add_argument("--data_root", type=str, default="data/oxford_pet")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["classification", "localization", "segmentation", "multitask"]
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_classes", type=int, default=37)
    parser.add_argument("--seg_classes", type=int, default=3)
    parser.add_argument("--dropout_p", type=float, default=0.5)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = get_device()
    print("Using device:", device)

    train_loader, val_loader = create_dataloaders(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size
    )

    model = build_model(
        task=args.task,
        num_classes=args.num_classes,
        seg_classes=args.seg_classes,
        dropout_p=args.dropout_p
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    cls_loss_fn = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    iou_loss = IoULoss(reduction="mean")
    seg_loss_fn = nn.CrossEntropyLoss()

    wandb.init(
        project="da6401-assignment2",
        name=f"{args.task}-run",
        config=vars(args)
    )

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        if args.task == "classification":
            train_loss, train_acc = train_one_epoch_classification(
                model, train_loader, optimizer, cls_loss_fn, device
            )
            val_loss, val_acc = validate_classification(
                model, val_loader, cls_loss_fn, device
            )

            print(
                f"Epoch [{epoch+1}/{args.epochs}] "
                f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}"
            )

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            })

        elif args.task == "localization":
            train_loss = train_one_epoch_localization(
                model, train_loader, optimizer, mse_loss, iou_loss, device
            )
            val_loss = validate_localization(
                model, val_loader, mse_loss, iou_loss, device
            )

            print(
                f"Epoch [{epoch+1}/{args.epochs}] "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
            })

        elif args.task == "segmentation":
            train_loss = train_one_epoch_segmentation(
                model, train_loader, optimizer, seg_loss_fn, device
            )
            val_loss = validate_segmentation(
                model, val_loader, seg_loss_fn, device
            )

            print(
                f"Epoch [{epoch+1}/{args.epochs}] "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
            })

        else:  # multitask
            train_loss = train_one_epoch_multitask(
                model, train_loader, optimizer,
                cls_loss_fn, mse_loss, iou_loss, seg_loss_fn, device
            )
            val_loss = validate_multitask(
                model, val_loader,
                cls_loss_fn, mse_loss, iou_loss, seg_loss_fn, device
            )

            print(
                f"Epoch [{epoch+1}/{args.epochs}] "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
            })

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            if args.task == "classification":
                save_path = os.path.join(args.save_dir, "classifier.pth")
            elif args.task == "localization":
                save_path = os.path.join(args.save_dir, "localizer.pth")
            elif args.task == "segmentation":
                save_path = os.path.join(args.save_dir, "unet.pth")
            else:
                save_path = os.path.join(args.save_dir, "multitask.pth")

            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

    wandb.finish()


if __name__ == "__main__":
    main()