# Project Notes

Talon learns from markdown files stored in a knowledge directory.

## Goals

1. Read markdown documents recursively.
2. Build a vocabulary from the dataset.
3. Train a decoder-only Transformer with PyTorch.
4. Save a reusable checkpoint for later text generation.

## Training idea

The model predicts the next token in a sequence.
Training examples come from chunks of markdown text.
