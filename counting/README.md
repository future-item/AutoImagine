# Counting Framework

## Quick Start

**Download dataset**:

From our huggingface repo:

```bash
cd placing
git clone https://huggingface.co/datasets/future-item/autoimagine-counting
mv autoimagine-counting/* .
rmdir autoimagine-counting
```

**Set environment variables**:

```bash
# For OpenRouter API (default: https://openrouter.ai/api/v1)
export OPENROUTER_API_KEY="your-openrouter-api-key"

# For local LLM server (default: http://localhost:8010/v1)
export LOCAL_API_KEY="your-local-api-key"
```

**Run Counting tasks**:

To run a single question in a single task:

```bash
python run-counting.py 
```

To run the entire dataset:

```bash
python run-counting.py all
```
