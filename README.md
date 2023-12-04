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
Uniting all the texts in single file to train Sentencepiece tokenizer, ```tokenize_dataset.py``` to create .npy dataset
```bash
python concat_stories.py
python tokenize_dataset.py
```

## Training
If not running previous scripts put ```tiny_stories_tokenized.npy``` from Yandex.Disk into the main directory before running
```bash
python train.py -wk="YOUR_WANDB_KEY"
```

## Testing
Put ```checkpoint-epoch12.pth``` from Yandex.Disk into the main directory, launch this to get PPL (uses ```evaluate``` which is not installed by default)
```bash
python test.py
```