import numpy as np
from glob import glob
import json
from tqdm import tqdm
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer


if __name__ == "__main__":
    tokenizer = SentencePieceProcessor(model_file="MurkyLM.model")
    data = np.load("tiny_stories_tokenized.npy")
    print(tokenizer.decode(data[0].tolist()))
