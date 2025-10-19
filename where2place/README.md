# Where2Place: Object Placement Framework

## Quick Start

**Download dataset**:

Download two folders:

- prompt (i.e. prompts for this algorithm)
- dataset (i.e. the altered where2place dataset)

from [our huggingface repo](https://huggingface.co/datasets/future-item/autoimagine-w2p).

```bash
cd where2place
git clone https://huggingface.co/datasets/future-item/autoimagine-w2p
mv autoimagine-w2p/prompt autoimagine-w2p/dataset .
rm -rf autoimagine-w2p
```

**Set environment variables**:

```bash
# For OpenRouter API (default: https://openrouter.ai/api/v1)
export OPENROUTER_API_KEY="your-openrouter-api-key"

# For local LLM server (default: http://localhost:8010/v1)
export LOCAL_API_KEY="your-local-api-key"
```

**Run Where2Place tasks**:
```bash
# Let's run the 33-th and 66-th sample in parallel!
python run_parallel-placer.py --baseline ours --use_model "model-name" --workers 2 --specify-id 33 66 # Run the entire dataset one by one
```

**Evaluate results**:
```bash
python check_masks.py --batch-check
```

