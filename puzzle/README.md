# Puzzle Placement Framework

## Quick Start

**Download dataset**:

Download two folders:

- data (i.e. prompts for this algorithm)
- dataset (i.e. the puzzles)

from [our huggingface repo](https://huggingface.co/datasets/future-item/autoimagine-puzzle).

```bash
cd puzzle
git clone https://huggingface.co/datasets/future-item/autoimagine-puzzle
mv autoimagine-puzzle/data autoimagine-puzzle/dataset .
rm -rf autoimagine-puzzle
```

**Set environment variables**:

```bash
# For OpenRouter API (default: https://openrouter.ai/api/v1)
export OPENROUTER_API_KEY="your-openrouter-api-key"

# For local LLM server (default: http://localhost:8010/v1)
export LOCAL_API_KEY="your-local-api-key"
```

**Run Puzzle tasks**:

```bash
python run-puzzle.py --baseline ours --model openai/gpt-4o --workers 1 --tasks 0
```

Evaluate results:
```bash
python calc_stats.py
```