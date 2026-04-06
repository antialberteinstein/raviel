from config import vocab_size
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

class Tokenizer:
    def __init__(self, file_path="vi_tokenizer.json"):
        self.tokenizer = HFTokenizer.from_file(file_path)

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)


class TokenizerTrainer:
    def __init__(self, train_files, out_path="vi_tokenizer.json"):
        # Initialize BPE tokenizer and whitespace pre-tokenizer.
        self.tokenizer = HFTokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()

        self.trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=[
                "[PAD]",
                "[UNK]",
                "[BOS]",
                "[EOS]",
            ],  # Các token đặc biệt bắt buộc phải có.
            min_frequency=2,  # Only keep tokens that appear at least twice.
        )

        self.train_files = train_files
        self.out_path = out_path

    def train(self):
        # Train and save tokenizer.
        self.tokenizer.train(self.train_files, self.trainer)

        self.tokenizer.save(self.out_path)

        print(f"Đã lưu tokenizer vào file {self.out_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2:
        first_arg = sys.argv[1]
        second_arg = sys.argv[2]

        if first_arg == "--train":
            trainer = TokenizerTrainer(train_files=[second_arg])
            trainer.train()
        else:
            print("Please run with --train input_file to start training")
    else:
        print("Please run with --train input_file to start training")
