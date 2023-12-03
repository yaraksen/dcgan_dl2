# Big homework 1. BLMs (Boutique LMs)
## Aksenov Yaroslav

### Disclaimer: some functions and classes were heavily based on my DLA template

## Data
The dataset can be loaded from HuggingFace and opened with tar:
```bash
wget --quiet --show-progress "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
tar -xvf TinyStories_all_data.tar.gz -C tiny_stories
```

## Data manipulation
Uniting all the texts in single file to train Sentencepiece tokenizer
```python
python concat_stories.py
```

## Training
```python
python train.py
```