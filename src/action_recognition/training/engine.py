from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def run_epoch(model, loader: DataLoader, criterion, device, optimizer=None) -> tuple[float, float]:
    """One pass over `loader`. Pass `optimizer` to train, omit it to evaluate."""
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.enable_grad() if train_mode else torch.no_grad():
        for clips, labels in tqdm(loader, leave=False):
            clips, labels = clips.to(device), labels.to(device)

            if train_mode:
                optimizer.zero_grad()
            logits = model(clips)
            loss = criterion(logits, labels)
            if train_mode:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_count += labels.size(0)

    return total_loss / total_count, total_correct / total_count
