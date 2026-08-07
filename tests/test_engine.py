import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from action_recognition.training.engine import run_epoch


class _TinyClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x):
        return self.fc(x)


def _make_loader(n=8, num_classes=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 4, generator=g)
    y = torch.randint(0, num_classes, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=4)


def test_run_epoch_train_mode_updates_weights_and_returns_valid_metrics():
    torch.manual_seed(0)
    model = _TinyClassifier()
    loader = _make_loader()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    before = model.fc.weight.clone()
    loss, acc = run_epoch(model, loader, criterion, torch.device("cpu"), optimizer)

    assert model.training is True
    assert loss > 0
    assert 0.0 <= acc <= 1.0
    assert not torch.equal(before, model.fc.weight)  # a training step happened


def test_run_epoch_eval_mode_leaves_weights_untouched():
    torch.manual_seed(0)
    model = _TinyClassifier()
    loader = _make_loader()
    criterion = nn.CrossEntropyLoss()

    before = model.fc.weight.clone()
    loss, acc = run_epoch(model, loader, criterion, torch.device("cpu"))

    assert model.training is False
    assert torch.equal(before, model.fc.weight)  # no optimizer -> no update
    assert loss > 0
    assert 0.0 <= acc <= 1.0


def test_run_epoch_perfect_predictions_yield_accuracy_one():
    # A model that just echoes its input as logits; feeding it scaled one-hot
    # vectors of the true label makes every prediction correct by construction.
    class _Echo(nn.Module):
        def forward(self, x):
            return x

    labels = torch.tensor([0, 1, 0, 1])
    onehot = F.one_hot(labels, num_classes=2).float() * 10
    loader = DataLoader(TensorDataset(onehot, labels), batch_size=2)

    loss, acc = run_epoch(_Echo(), loader, nn.CrossEntropyLoss(), torch.device("cpu"))

    assert acc == 1.0
    assert loss >= 0
