"""
Configuration objects for Talon's training and inference runs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass
class TalonConfig:
    # These defaults keep the starter model small enough to train on a consumer GPU.
    model_name: str = "talon-base"
    data_dir: str = "data/knowledge"
    output_dir: str = "artifacts/talon-base"
    block_size: int = 128
    batch_size: int = 16
    train_split: float = 0.9
    max_steps: int = 500
    eval_interval: int = 50
    eval_batches: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    dropout: float = 0.1
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    seed: int = 42
    device: str = "cpu"
    vocab_size: int = 0

    def save_json(self, path: str | Path) -> None:
        destination = Path(path)
        # Store config as plain JSON so checkpoints stay easy to inspect and edit.
        destination.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "TalonConfig":
        source = Path(path)
        return cls(**json.loads(source.read_text(encoding="utf-8")))
