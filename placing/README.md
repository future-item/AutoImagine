# Placing Framework

## Quick Start

**Download dataset**:

From our huggingface repo:

```bash
cd placing
git clone https://huggingface.co/datasets/future-item/autoimagine-3dgs
mv autoimagine-3dgs/* .
rmdir autoimagine-3dgs
```

**Set environment variables**:

```bash
# For OpenRouter API (default: https://openrouter.ai/api/v1)
export OPENROUTER_API_KEY="your-openrouter-api-key"

# For local LLM server (default: http://localhost:8010/v1)
export LOCAL_API_KEY="your-local-api-key"
```

**Run Placing tasks**:

To run a single question in a single task (in our example code, it's the first question of task `classroom-2`):

```bash
bash run-placing-single.sh
```

To run the entire dataset (i.e. every question in every task):

```bash
bash run-placing.sh
```
