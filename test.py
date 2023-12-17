from tqdm import tqdm
from sentencepiece import SentencePieceProcessor
import torch
from model import MurkyLM


@torch.no_grad()
def generate(model, eos_id: int, device, N: int, temp: float = 1.0, max_len=256):
    model.eval()
    input = torch.full((N, 1), fill_value=eos_id).to(device)
    for _ in range(max_len - 1):
        logits = torch.nn.functional.softmax(model(input)[:, -1, :] / temp, dim=-1)
        input = torch.cat((input, torch.multinomial(logits, 1)), dim=-1)
    return input


def cut_after_eos(input_ids: torch.Tensor, eos_id):
    eos_idx = torch.nonzero(input_ids == eos_id)
    if eos_idx.shape[0] > 0:
        return input_ids[: eos_idx[0]]
    else:
        return input_ids


def murkylm_ppl(pretrained_path: str, N: int, temp: float):
    from evaluate import load

    device_id = 0
    device = torch.device(f"cuda:{device_id}")
    print(device)
    torch.cuda.set_device(device)
    max_len = 256
    vocab_size = 5000
    model_params = {
        "d_model": 512,
        "nhead": 8,
        "d_hid": 2048,
        "nlayers": 8,
        "dropout": 0.05,
        "max_len": max_len,
        "device": device,
    }
    ##### END CONFIG ######

    tokenizer = SentencePieceProcessor(model_file="MurkyLM.model")
    perplexity = load("perplexity", module_type="metric")

    model = MurkyLM(vocab_size, **model_params)
    checkpoint = torch.load(pretrained_path, device)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)

    predictions = []
    lens = []
    for i in tqdm(range(1), desc="Generating..."):
        encoded_texts = generate(
            model, tokenizer.eos_id(), device, N=N, temp=temp, max_len=256
        )
        for enc in encoded_texts:
            enc = cut_after_eos(enc[1:], tokenizer.eos_id())
            lens.append(len(enc))
            predictions.append(tokenizer.decode(enc.tolist()))
    results = perplexity.compute(
        predictions=predictions, batch_size=8, model_id="gpt2-large", device="cuda"
    )
    print(lens)
    print("\n\n".join(predictions[:5]))
    print(f"mean ppl for temp={temp}:", results["mean_perplexity"])


if __name__ == "__main__":
    ppl, texts = murkylm_ppl("checkpoint-epoch12.pth")
    print("mean_perplexity:", ppl)
