# Big homework 2. Image generation (DCGAN)
## Aksenov Yaroslav

### Disclaimer: some functions and classes were heavily based on my DLA template

## Data
The dataset can be loaded from Kaggle and put ```cats``` dir into ```data/```, so ```data/cats/*.jpg```:
```bash
kaggle datasets download -d spandan2/cats-faces-64x64-for-generative-models
unzip cats-faces-64x64-for-generative-models
```

## Training
```bash
python train.py -wk="YOUR_WANDB_KEY"
```

## Testing
Checkpoint ```checkpoint-epoch1900.pth``` is on GitHub and should be in the main directory
```bash
python test.py
```