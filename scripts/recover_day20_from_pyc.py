from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
import multiprocessing


def load_module(root: pathlib.Path, name: str, relpath: pathlib.Path):
    path = root / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    import src  # noqa: F401

    pkg = types.ModuleType("src.training")
    pkg.__path__ = [str(root / "src" / "training")]
    pkg.__package__ = "src.training"
    sys.modules["src.training"] = pkg

    load_module(
        root,
        "src.training.detector_supervised_utils",
        pathlib.Path("src/training/__pycache__/detector_supervised_utils.cpython-313.pyc"),
    )
    mod = load_module(
        root,
        "src.training.train_detector_supervised",
        pathlib.Path("src/training/__pycache__/train_detector_supervised.cpython-313.pyc"),
    )

    sys.argv = [
        "train_detector_supervised",
        "--train_path", str(root / "data/processed/mimiciv_train_detector_supervised.pkl"),
        "--val_path", str(root / "data/processed/mimiciv_val_detector_supervised.pkl"),
        "--out_dir", str(root / "outputs/detector/day20_supervised/run_luxury"),
        "--num_workers", "0",
    ]
    mod.main()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
