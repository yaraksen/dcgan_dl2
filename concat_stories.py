import numpy as np
from glob import glob
import json
from tqdm import tqdm

def main(data_path: str, out_file: str):
    files = glob(f"{data_path}/*.json")
    all_texts = ""
    for file in tqdm(files, desc="Loading data..."):
        with open(file, "r") as f:
            stories = [item["story"] for item in json.loads(f.read())]
            all_texts += "".join(stories)
    
    with open(out_file, "w") as f:
        f.write(all_texts)

if __name__ == "__main__":
    main("tiny_stories", "all_stories.txt")