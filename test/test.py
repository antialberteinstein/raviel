import sys

sys.path.append("src")

import argparse

import torch

from config.config import block_size, device
from model.transformer_model import LargeLanguageModel
from tokenizer.tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/model.pt")
    parser.add_argument("--tokenizer", default="models/vi_tokenizer.json")
    parser.add_argument("--prompt_file", default="dataset/input.txt")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--out", default="models/generated.txt")
    args = parser.parse_args()

    tokenizer = Tokenizer(args.tokenizer)

    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read().strip()

    prompt_ids = tokenizer.encode(prompt)
    if len(prompt_ids) == 0:
        prompt_ids = [0]

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    model = LargeLanguageModel().to(device)
    ckpt = torch.load(args.model, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        generated = model.generate(idx[:, -block_size:], args.max_new_tokens)

    text = tokenizer.decode(generated[0].tolist())

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Wrote output to {args.out}")


if __name__ == "__main__":
    main()
