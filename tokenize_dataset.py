import numpy as np
from glob import glob
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

def main(data_path: str, tokenizer_tag: str, max_len: int):
    files = glob(f"{data_path}/*.json")
    tokenizer = SentencePieceProcessor(model_file=f"{tokenizer_tag}.model")

    all_input_ids = []
    for file in tqdm(files, desc="Tokenizing and stacking texts..."):
        with open(file, "r") as f:
            for item in json.loads(f.read()):
                all_input_ids += tokenizer.encode(item["story"]) + [tokenizer.eos_id()]
    
    print("Token number:", len(all_input_ids))
    dataset = []
    for start_idx in range(0, len(all_input_ids) - max_len, max_len):
        dataset.append(all_input_ids[start_idx:start_idx + max_len])
    
    print("Dataset size:", len(dataset))
    np.save("tiny_stories_tokenized.npy", np.array(dataset, dtype=np.int16))

if __name__ == "__main__":
    main(data_path="tiny_stories",
         tokenizer_tag="MurkyLM",
         max_len=256)