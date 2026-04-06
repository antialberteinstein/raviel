import argparse
import torch

from config import block_size, device
from dataloader import DataLoader
from tokenizer import Tokenizer
from transformer_model import LargeLanguageModel


def estimate_loss(model, data_loader, eval_iters):
    model.eval()
    out = {}
    with torch.no_grad():
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                x, y = data_loader.get_batch(split)
                _, loss = model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
    model.train()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="input.txt")
    parser.add_argument("--tokenizer", default="vi_tokenizer.json")
    parser.add_argument("--out", default="model.pt")
    parser.add_argument("--max_iters", type=int, default=2000)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--eval_iters", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    args = parser.parse_args()

    tokenizer = Tokenizer(args.tokenizer)
    data_loader = DataLoader(tokenizer, args.input)

    model = LargeLanguageModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    print(f"Device: {device}")
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M")

    for step in range(args.max_iters):
        if step % args.eval_interval == 0 or step == args.max_iters - 1:
            losses = estimate_loss(model, data_loader, args.eval_iters)
            print(
                f"step {step}: train {losses['train']:.4f}, val {losses['val']:.4f}"
            )

        x, y = data_loader.get_batch("train")
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save({"model_state_dict": model.state_dict()}, args.out)
    print(f"Saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
