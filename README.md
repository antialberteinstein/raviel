# Raviel

Mau noi cho du an train mo hinh Transformer nho cho van ban tieng Viet. Du an gom 3 buoc chinh: train tokenizer, train model, va sinh van ban.

## Yeu cau
- Python 3.9+ (khuyen nghi)
- PyTorch
- Hugging Face `tokenizers`

Cai dat nhanh:
```bash
pip install torch tokenizers
```

## 1) Train tokenizer (BPE)
Neu ban chua co tokenizer, tao tu file text goc.
```bash
python tokenizer.py --train input.txt
```
Lenh nay tao ra file `vi_tokenizer.json` (mac dinh).

## 2) Train model
```bash
python train.py \
  --input input.txt \
  --tokenizer vi_tokenizer.json \
  --out model.pt \
  --max_iters 2000 \
  --eval_interval 200 \
  --eval_iters 100 \
  --learning_rate 3e-4
```
Sau khi train xong, model duoc luu tai `model.pt`.

## 3) Sinh van ban
```bash
python test.py \
  --model model.pt \
  --tokenizer vi_tokenizer.json \
  --prompt_file input.txt \
  --max_new_tokens 100 \
  --out generated.txt
```
Ket qua duoc ghi vao `generated.txt`.

## Tuy chinh cau hinh
Cac tham so co ban nam trong `config.py`:
- `batch_size`: so luong mau moi batch
- `block_size`: do dai ngu canh
- `n_embd`: kich thuoc embedding
- `vocab_size`: kich thuoc tu vung
- `n_layer`: so lop Transformer block

## Ghi chu
- Thiet bi se tu dong chon CUDA, MPS (Apple Silicon), hoac CPU trong [gpu_support.py](gpu_support.py).
- Neu prompt rong, he thong se dung token `[PAD]` de khoi tao sinh van ban.
