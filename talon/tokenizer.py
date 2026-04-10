"""
A tiny character-level tokenizer used by Talon's base model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json


SPECIAL_TOKENS = ["<pad>", "<eos>", "<unk>"]


@dataclass
class CharTokenizer:
    stoi: dict[str, int]
    itos: list[str]

    @classmethod
    def fit(cls, texts: list[str]) -> "CharTokenizer":
        # The starter model works at the character level to keep training simple.
        charset = sorted({character for text in texts for character in text})
        itos = [*SPECIAL_TOKENS, *charset]
        stoi = {token: index for index, token in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    @property
    def pad_id(self) -> int:
        return self.stoi["<pad>"]

    @property
    def eos_id(self) -> int:
        return self.stoi["<eos>"]

    @property
    def unk_id(self) -> int:
        return self.stoi["<unk>"]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(character, self.unk_id) for character in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        characters: list[str] = []
        for token_id in token_ids:
            token = self.itos[token_id]
            if skip_special_tokens and token in SPECIAL_TOKENS:
                continue
            characters.append("?" if token == "<unk>" else token)
        return "".join(characters)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        # Only the index-to-token list is needed; stoi can be rebuilt during load.
        payload = {"itos": self.itos}
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        itos = payload["itos"]
        stoi = {token: index for index, token in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)
