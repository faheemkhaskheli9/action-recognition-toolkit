import pandas as pd

from action_recognition.scripts.build_manifest import main as build_manifest_main
from action_recognition.scripts.split_dataset import main as split_dataset_main


def test_build_manifest_main_writes_a_manifest_csv(class_folder_dataset, tmp_path, monkeypatch, capsys):
    output = tmp_path / "manifest.csv"
    monkeypatch.setattr("sys.argv", ["ar-build-manifest", str(class_folder_dataset), str(output)])

    build_manifest_main()

    df = pd.read_csv(output)
    assert len(df) == 5
    assert set(df["label"]) == {"jump", "wave"}
    assert "Wrote 5 rows across 2 classes" in capsys.readouterr().out


def test_split_dataset_main_adds_split_column_and_writes_label_map(tmp_path, monkeypatch, capsys):
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "video_path": [f"/v{i}.mp4" for i in range(10)],
            "label": ["jump"] * 5 + ["wave"] * 5,
        }
    ).to_csv(manifest_path, index=False)

    monkeypatch.setattr(
        "sys.argv",
        ["ar-split-dataset", str(manifest_path), "--val-frac", "0.2", "--test-frac", "0.2", "--seed", "0"],
    )

    split_dataset_main()

    df = pd.read_csv(manifest_path)
    assert set(df["split"]) <= {"train", "val", "test"}
    assert len(df) == 10

    label_map_path = manifest_path.with_name("label_map.json")
    assert label_map_path.exists()
    assert "Wrote" in capsys.readouterr().out


def test_split_dataset_main_respects_explicit_output_path(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame({"video_path": ["/a.mp4", "/b.mp4"], "label": ["jump", "wave"]}).to_csv(manifest_path, index=False)
    output_path = tmp_path / "split_manifest.csv"

    monkeypatch.setattr(
        "sys.argv",
        ["ar-split-dataset", str(manifest_path), "--output", str(output_path), "--val-frac", "0.0", "--test-frac", "0.0"],
    )

    split_dataset_main()

    assert output_path.exists()
    assert (output_path.with_name("label_map.json")).exists()
    # the original manifest (without --output) is left untouched
    assert "split" not in pd.read_csv(manifest_path).columns
