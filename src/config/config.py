from config import gpu_support

batch_size = 32  # Số lượng đoạn văn xử lý song song.
block_size = 8  # Độ dài ngữ cảnh.

device = gpu_support.get_device()

n_head = 4
n_embd = 32  # Số chiều của vector embedding.
vocab_size = 5000

# Số khối Transformer Block xếp chồng
n_layer = 10
