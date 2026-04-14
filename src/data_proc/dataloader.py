import torch

from config.config import batch_size, block_size, device


class DataLoader:
    def __init__(
        self,
        tokenizer,
        input_file,  # File đầu vào
        test_ratio=0.1,  # Tỷ lệ tập test.
    ) -> None:
        # Load and encode input data.
        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read()

        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

        # Split train/val.
        n = int((1 - test_ratio) * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def get_batch(self, split):
        # Pick source split.
        if split == "train":
            data_source = self.train_data
        else:
            data_source = self.val_data

        # Sample starting positions (avoid overflow beyond block_size).
        ix = torch.randint(len(data_source) - block_size, (batch_size,))

        # Slice input/target sequences.
        x = torch.stack([data_source[i : i + block_size] for i in ix])
        y = torch.stack([data_source[i + 1 : i + block_size + 1] for i in ix])

        # Move to device.
        x, y = x.to(device), y.to(device)

        return x, y
