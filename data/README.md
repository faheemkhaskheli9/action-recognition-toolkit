# data/

This directory is where you put your own video files. It is gitignored —
nothing under here is committed. Two common layouts:

## 1. Unlabeled videos, to label with the web app

```
data/raw/*.mp4
```

Point the Django web app's Label page (`webapp/`) at `data/raw` and it will
write labels to `data/manifest.csv` as you go.

## 2. Already organized by class folder

```
data/raw/<class_name>/*.mp4
```

Run `python -m action_recognition.scripts.build_manifest data/raw data/manifest.csv`
to generate a manifest directly, skipping the labeling app.

Either way, once you have `data/manifest.csv`, run
`python -m action_recognition.scripts.split_dataset data/manifest.csv` to add
train/val/test splits, then start training — see the root README.
