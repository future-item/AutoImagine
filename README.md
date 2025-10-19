# Auto Imagine
Auto Imagine is a research-oriented project integrating multiple tasks on visual reasoning and spatial understanding.
It consists of four independent modules, each with its own documentation and experimental setup.

## Overview

- placing – Object placement and spatial arrangement reasoning.

- counting – Object counting.

- puzzle – Visual and logical puzzle-solving tasks.

- where2place – Context-aware object placement prediction.

## Installing the environment

From our huggingface repo:

```bash
cd placing
git clone https://huggingface.co/datasets/future-item/autoimagine-env
mv autoimagine-env/* .
rmdir autoimagine-env
```

```bash
ln -s ../dataset gaussian-grouping/data # Create symbolic link from dataset to gaussian-grouping/data
ln -s ../dataset gaussian-grouping/output # Create symbolic link from dataset to gaussian-grouping/output
```

```bash
conda create -n auto-imagine python=3.10 -y
conda activate auto-imagine 
pip install -r requirements.txt
pip install -e segment-anything-2
pip install -e gaussian-grouping/submodules/simple-knn
pip install -e gaussian-grouping/submodules/diff-gaussian-rasterization
# Note: our diff-gaussian-rasterization module is an ALTERATION of https://github.com/graphdeco-inria/diff-gaussian-rasterization. DO NOT USE the original one.
```

## Getting Started

For detailed instructions on data preparation, training, and example usage, please refer to the individual README.md files within each subfolder.
