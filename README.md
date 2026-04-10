# Talon

Talon is a small GPT-style language model starter built with PyTorch. It learns from raw markdown files, trains on next-token prediction, and can generate new markdown-style text from a prompt.

## What is included

- A markdown corpus loader that reads `*.md` files recursively
- A simple character-level tokenizer learned from your markdown files
- A compact decoder-only Transformer model
- A training entrypoint that saves checkpoints and tokenizer metadata
- A generation entrypoint for sampling text from a trained checkpoint

## Project layout

```text
data/knowledge/      # Add your markdown files here
talon/               # Core Talon package
```

## Setup

Create or activate your virtual environment, then install PyTorch:

```powershell
.\.venv\Scripts\pip.exe install -r requirements.txt
```

## Train Talon

Talon reads all markdown files under `data/knowledge` by default.

```powershell
.\.venv\Scripts\python.exe -m talon.train
```

Useful options:

```powershell
.\.venv\Scripts\python.exe -m talon.train `
  --data-dir data/knowledge `
  --output-dir artifacts/talon-base `
  --max-steps 800 `
  --batch-size 16 `
  --block-size 128 `
  --n-layer 4 `
  --n-head 4 `
  --n-embd 128
```

## Generate text

```powershell
.\.venv\Scripts\python.exe -m talon.generate `
  --checkpoint-dir artifacts/talon-base `
  --prompt "# Talon`n`n"
```

## Answer with retrieval

Talon can retrieve relevant local markdown passages during answer time, and it can optionally pull live web pages too.

Local retrieval:

```powershell
.\.venv\Scripts\python.exe -m talon.answer `
  --checkpoint-dir artifacts/talon-base `
  --question "How should Talon answer users?" `
  --answer-length medium `
  --tone balanced `
  --style balanced `
  --explanation-level medium `
  --show-sources
```

Local plus live web retrieval:

```powershell
.\.venv\Scripts\python.exe -m talon.answer `
  --checkpoint-dir artifacts/talon-base `
  --question "What is Python used for?" `
  --answer-length short `
  --tone formal `
  --style logical `
  --explanation-level high `
  --web `
  --show-sources
```

Web retrieval is on by default. Use `--no-web` if you want a local-only answer.
Use `--answer-length short`, `--answer-length medium`, or `--answer-length long` to control how brief or detailed the answer should be.
Use `--tone casual`, `--tone balanced`, or `--tone formal` to control the style of the answer.
Use `--style balanced`, `--style logical`, or `--style creative` to control whether Talon sounds more neutral, more structured, or more imaginative.
Use `--explanation-level low`, `--explanation-level medium`, `--explanation-level high`, or `--explanation-level advanced` to control how simple or deep the explanation should be.

For small checkpoints, Talon automatically falls back to a readable extractive answer built from retrieved passages. This is much more useful than forcing a tiny model to generate from almost no context.

If you want generative retrieval to work better, retrain with a larger `--block-size` like `256` or `512`:

```powershell
.\.venv\Scripts\python.exe -m talon.train `
  --block-size 256 `
  --max-steps 1200 `
  --output-dir artifacts/talon-rag
```

Then answer with that checkpoint:

```powershell
.\.venv\Scripts\python.exe -m talon.answer `
  --checkpoint-dir artifacts/talon-rag `
  --question "How should Talon answer users?" `
  --web `
  --show-sources
```

## Interactive chat

Start an interactive session:

```powershell
.\.venv\Scripts\python.exe -m talon.chat `
  --checkpoint-dir artifacts/talon-base `
  --answer-length medium `
  --tone balanced `
  --style balanced `
  --explanation-level medium `
  --web `
  --show-sources
```

Web retrieval starts enabled by default in chat and in the desktop window. Use `--no-web` to disable it at launch.

Useful chat commands:

- `/help`
- `/length short`
- `/length medium`
- `/length long`
- `/tone casual`
- `/tone balanced`
- `/tone formal`
- `/style balanced`
- `/style logical`
- `/style creative`
- `/explain low`
- `/explain medium`
- `/explain high`
- `/explain advanced`
- `/quit`
- `/web on`
- `/web off`
- `/sources on`
- `/sources off`

The desktop window also has dropdowns for length, tone, style, and explanation level, so you can switch answer style without restarting Talon.

## Quick launchers

Open the small Talon window:

```powershell
.\run_talon.bat
```

Launch the window and pass extra flags through to `talon.gui`:

```powershell
.\run_talon.bat --extractive-only
.\run_talon.bat --checkpoint-dir artifacts/talon-base --web --show-sources --answer-length long --tone formal --style logical --explanation-level high
```

Train Talon with the default GPU-focused settings:

```powershell
.\train_talon.bat
```

Pass extra training flags through to `talon.train`:

```powershell
.\train_talon.bat --max-steps 2000 --batch-size 24
```

## GPU setup

Right now the project will use CUDA automatically if your installed PyTorch build supports it. If `torch.cuda.is_available()` is false even though you have an NVIDIA GPU, the usual cause is that the venv has a CPU-only PyTorch wheel installed.

PyTorch's official Windows instructions say to choose the Windows `pip` install with a CUDA build that matches your system, and they recommend verifying with `torch.cuda.is_available()` afterward. The official install pages are:

- [PyTorch Get Started](https://pytorch.org/get-started/locally/)
- [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/)

## Give Talon web-sourced knowledge

Talon can pull webpage text into markdown files so the content becomes part of the training set.

Fetch one page:

```powershell
.\.venv\Scripts\python.exe -m talon.fetch_web `
  --url "https://example.com" `
  --output-dir data/knowledge/web
```

Fetch multiple pages:

```powershell
.\.venv\Scripts\python.exe -m talon.fetch_web `
  --url "https://example.com" `
  --url "https://www.python.org"
```

Or fetch from a file that contains one URL per line:

```powershell
.\.venv\Scripts\python.exe -m talon.fetch_web `
  --from-file urls.txt
```

After fetching pages, retrain Talon so it learns from the new markdown files.

## Teaching Talon with markdown

Put markdown files in `data/knowledge/`. Talon trains on the raw text, so the shape and style of your markdown matter.

Good examples:

- Character background and personality files
- Project design notes
- Worldbuilding docs
- FAQ or product docs
- Conversation examples written in markdown

The starter uses a character-level tokenizer because it is easy to understand and works well for a first small model. Later, we can upgrade Talon to use a subword tokenizer, larger datasets, better evaluation, and instruction tuning.

## Important note

This starter does not give Talon full autonomous live browsing like a hosted assistant. It gives Talon a usable first step: importing web pages into markdown training data. If you want, the next upgrade can be live search and retrieval at answer time instead of only learning from fetched pages.
