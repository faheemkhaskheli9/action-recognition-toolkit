# Action Recognition for Beginners: From Video to Prediction

*A code-first introduction to how machines recognize human actions in video — using this repository as the running example.*

If you can train an image classifier, action recognition is the natural
next step: same idea, plus a new dimension — **time**. This article walks
through the concepts and the actual code that implements them, so by the
end you can point at a function and say "that's where X happens."

---

## 1. What makes this harder than image classification

An image classifier answers "what's in this picture?" from one static
frame. Action recognition answers "what is happening across these
frames?" — and that requires the model to see *motion*, not just objects.

Two clips can contain identical objects (a person, a chair) and still be
different actions ("sitting down" vs "standing up") purely because of the
order and speed of movement. So an action-recognition model needs two
things a plain image classifier doesn't:

1. **Spatial features** — what's in each frame (a person, their pose,
   their surroundings). A normal CNN is good at this.
2. **Temporal features** — how those things change from frame to frame.
   This is the genuinely new problem.

Almost every design decision in an action-recognition system — how you
sample frames, which architecture you pick, how you augment data — comes
back to how it handles that second part.

---

## 2. Turning a video file into model input

A video file is not usable as-is: clips are different lengths, different
frame rates, and far too many frames to feed a network at once (a 10
second clip at 30fps is 300 frames). The first job of the data pipeline
is to turn a variable-length video into a **fixed-size tensor**.

This repo does it with uniform sampling — pick `N` frames spread evenly
across the clip's duration, regardless of how long the clip is
([`dataset.py`](../src/action_recognition/data/dataset.py)):

```python
def _sample_indices(total_frames: int, num_frames: int) -> list[int]:
    if total_frames <= num_frames:
        return [min(i, total_frames - 1) for i in range(num_frames)]
    step = total_frames / num_frames
    return [int(step * i + step / 2) for i in range(num_frames)]
```

If a clip is shorter than `num_frames`, the last frame repeats to pad it
out. Otherwise, frames are picked at evenly spaced offsets — frame
`step/2`, `step + step/2`, `2*step + step/2`, and so on — so a 3-second
clip and a 10-second clip of the same action both end up as, say, 16
frames covering their whole duration.

That's a deliberate simplification worth understanding rather than
tweaking blindly: it means motion that happens in a tiny fraction of the
clip (a fast flinch) gets the same number of samples as motion spread
evenly across it, and there's no optical-flow or dedicated motion
signal — the model has to infer motion purely from how the sampled
frames differ. It's cheap and it works well enough for clip-level
labels, which is why it's the default here.

Each sampled frame then goes through a `torchvision` transform (resize,
crop, normalize — see [`transforms.py`](../src/action_recognition/data/transforms.py)),
and `VideoClipDataset.__getitem__` stacks them into one tensor:

```python
def __getitem__(self, idx: int):
    video_path, label = self.samples[idx]
    frames = read_clip_frames(video_path, self.num_frames)
    clip = torch.stack([self.transform(frame) for frame in frames], dim=0)
    return clip, self.label_map[label]
```

The result is a tensor shaped `(T, C, H, W)` — T frames, 3 color
channels, height, width. A `DataLoader` batches these into
`(B, T, C, H, W)`, which is the shape every model in this repo expects
as input. Keeping that shape as the shared contract (rather than letting
each architecture want its own layout) is what lets one `Dataset`
class serve every model — see §4.

---

## 3. Two ways to model time

Once you have a `(B, T, C, H, W)` tensor, there are two common families
of approach. This repo implements one of each, so you can compare them
directly.

### Approach A: CNN + RNN (`cnn_lstm`)

Run a normal 2D image CNN on *each frame independently* to get a feature
vector per frame, then feed the sequence of feature vectors into an
LSTM, which is built for sequential data. The final LSTM hidden state
summarizes "what happened across the clip" and a linear layer maps that
to class logits.

[`models/cnn_lstm.py`](../src/action_recognition/models/cnn_lstm.py):

```python
@register_model("cnn_lstm")
class CNNLSTM(nn.Module):
    def __init__(self, num_classes, hidden_size=256, num_layers=1,
                 pretrained_backbone=True, freeze_backbone=False):
        super().__init__()
        backbone = torchvision.models.resnet18(weights=...)
        backbone.fc = nn.Identity()          # strip the 1000-class ImageNet head
        self.backbone = backbone
        self.lstm = nn.LSTM(feature_dim, hidden_size, num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        b, t, c, h, w = x.shape
        # collapse (B, T) into one big batch so the CNN runs on every
        # frame of every clip in one pass, then split them back apart
        features = self.backbone(x.reshape(b * t, c, h, w)).reshape(b, t, -1)
        output, _ = self.lstm(features)
        last_step = output[:, -1, :]          # only the final timestep's summary
        return self.classifier(last_step)
```

The `x.reshape(b * t, c, h, w)` line is the trick worth internalizing: a
2D CNN has no notion of "batch of clips," only "batch of images," so the
time dimension is temporarily folded into the batch dimension, run
through the CNN as `b*t` independent images, then unfolded back into
`(b, t, features)` so the LSTM can see it as a sequence again.

Because `resnet18` starts from ImageNet-pretrained weights, this
architecture is cheap to train from scratch on a small custom dataset —
only the LSTM and classifier head are learning from your data; the
backbone already knows how to see.

### Approach B: 3D CNN (`r3d18`)

Instead of treating frames independently and reasoning about time
*after* feature extraction, use convolutional kernels that slide over
time *as well as* space — a `3×3×3` kernel instead of `3×3`. Every layer
sees a few frames at once, so short-range motion (which way is that arm
moving, this instant) is captured directly by the convolution, not
bolted on afterward.

[`models/r3d.py`](../src/action_recognition/models/r3d.py):

```python
@register_model("r3d18")
class R3D18(nn.Module):
    def __init__(self, num_classes, pretrained=True, freeze_backbone=False):
        super().__init__()
        net = torchvision.models.video.r3d_18(weights=...)   # Kinetics-pretrained
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        self.net = net

    def forward(self, x):
        # dataset yields (B, T, C, H, W); r3d_18 wants (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        return self.net(x)
```

Note the `permute` — this is the one place the two architectures
disagree about tensor layout, and it's handled *inside* the model, not
the dataset, so `VideoClipDataset` stays architecture-agnostic. That's a
useful pattern in general: keep your data pipeline's output shape fixed
and let each consumer adapt it, rather than parameterizing the pipeline
per-consumer.

`r3d18` is pretrained on Kinetics (an action-recognition dataset), so
unlike `cnn_lstm`'s ImageNet backbone, its pretraining already
understands motion, not just static objects — often a stronger starting
point if your action classes resemble everyday human activities.

### Both share one contract

Nothing else in the codebase needs to know which of these you're using,
because both obey the same interface: `forward()` takes
`(B, T, C, H, W)` and returns `(B, num_classes)` logits. That's enforced
by nothing more than convention plus a registry (next section) — it's
the whole reason training, evaluation, and inference code don't have any
`if model_name == ...` branches anywhere.

---

## 4. The registry pattern: adding a model without touching the rest of the code

[`models/registry.py`](../src/action_recognition/models/registry.py) is
a ~30-line dictionary wrapped in two functions:

```python
_REGISTRY: dict[str, Callable] = {}

def register_model(name: str):
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator

def build_model(name: str, num_classes: int, **kwargs):
    return _REGISTRY[name](num_classes=num_classes, **kwargs)
```

`@register_model("cnn_lstm")` on the class is what makes the string
`"cnn_lstm"` in a YAML config resolve to the `CNNLSTM` class at
`build_model()` time. This is a common pattern worth learning
independent of action recognition: it decouples "what architectures
exist" from "which one a given training run uses," so a config file can
select a model by name instead of the codebase needing a big switch
statement. Registration only happens if the module has actually been
`import`-ed somewhere (`models/__init__.py` imports `cnn_lstm` and
`r3d` on package load) — it's an import-time side effect, not
auto-discovery, so a new architecture file that's never imported simply
won't show up.

---

## 5. The rest of the pipeline: labels, splits, training

Before any of the above runs, raw video files need to become labeled,
split data:

- **Manifest** — a CSV of `video_path, label` rows. It's the single
  source of truth for "which video is which class"
  ([`data/manifest.py`](../src/action_recognition/data/manifest.py)).
- **Split** — a stratified train/val/test split is added as a `split`
  column, one label at a time, so every class is represented in every
  split (rather than, say, all of class "jump" accidentally landing in
  `train` and none in `val`).
- **Label map** — a `{label: int}` dict (`build_label_map` sorts labels
  alphabetically and assigns indices), saved to `label_map.json` so
  inference later knows how to turn a predicted index back into a class
  name.

The training loop itself is the same shape you'd write for image
classification — because by the time you're inside the loop, a clip
*is* just a tensor:

```python
def run_epoch(model, loader, criterion, device, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    with torch.enable_grad() if train_mode else torch.no_grad():
        for clips, labels in loader:
            clips, labels = clips.to(device), labels.to(device)
            if train_mode:
                optimizer.zero_grad()
            logits = model(clips)
            loss = criterion(logits, labels)
            if train_mode:
                loss.backward()
                optimizer.step()
```

This is the entire idea behind "action recognition is image
classification plus a time dimension" made concrete: everything upstream
(sampling frames, stacking them, running them through a
time-aware architecture) exists to produce a `(B, num_classes)` logits
tensor, at which point it's ordinary cross-entropy classification.

---

## 6. Trying it yourself

This repo's CLI mirrors the pipeline above one command per stage:

```bash
# 1. (raw multi-person footage only) detect + track + crop into per-person clips
ar-extract-tracks --config configs/tracking/default.yaml

# 2. build data/manifest.csv from a <root>/<class_name>/<video> folder layout
ar-build-manifest --root data/raw --out data/manifest.csv

# 3. add a stratified train/val/test split + write data/label_map.json
ar-split-dataset --manifest data/manifest.csv

# 4. train (config controls model choice, frames per clip, image size, epochs, ...)
ar-train --config configs/default.yaml

# 5. classify a single pre-trimmed clip
ar-predict --checkpoint runs/exp1/best.pt --video path/to/clip.mp4
```

Swapping architectures is a one-line config change —
`model.name: cnn_lstm` → `model.name: r3d18` in `configs/default.yaml` —
because of the registry pattern in §4. Everything else (data loading,
training loop, checkpointing) stays identical.

---

## 7. Pitfalls worth knowing before you hit them

- **Small datasets overfit fast.** Both models here lean on pretrained
  backbones specifically so you're only training a small head on your
  data — starting from `pretrained: false` on a 50-clip dataset will
  mostly memorize it.
- **Uniform sampling can miss brief motion.** If your action is a quick
  gesture inside a long clip, most of your 16 sampled frames may land on
  the "before" and "after," not the action itself. Trimming clips
  tighter around the action helps more than changing the sampling code.
- **One label per clip, one clip per video.** This pipeline doesn't
  split a long video into multiple training clips or support multiple
  labels per clip — if your source footage is long and multi-event,
  trim it into one clip per action before building the manifest, or use
  `ar-extract-tracks` if the issue is *multiple people* rather than
  multiple events.
- **Checkpoints are self-contained.** `save_checkpoint` bundles model
  weights, `label_map`, and the training config into one `.pt` file —
  you never need to keep the manifest or config around to run inference
  later, which is a useful property to preserve if you extend the
  checkpoint format.

---

## Where to go from here

- Read `configs/default.yaml` top to bottom — every field is documented
  inline and it's the fastest map of what's tunable.
- Try both `cnn_lstm` and `r3d18` on the same small dataset and compare
  — it's the quickest way to build intuition for the CNN+RNN vs 3D-CNN
  trade-off beyond what's on the page.
- Once single-clip classification feels familiar, look at
  `inference/scene_predict.py` for how detection + tracking + per-track
  classification compose to handle raw, multi-person video.
