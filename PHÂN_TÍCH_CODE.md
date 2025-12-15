# 📊 PHÂN TÍCH CẤU TRÚC & KIỂM TRA CODE TRAINING

## I. CẤU TRÚC TỔNG QUAN NOTEBOOK

Notebook được chia thành **17 phần chính**:

```
┌─────────────────────────────────────────────┐
│  PHẦN 1: CÀI ĐẶT (Cell 4)                  │
│  • Import thư viện (torch, spacy, tqdm...)  │
│  • Kiểm tra device (CPU/GPU)                │
└─────────────────────────────────────────────┘
           ⬇
┌─────────────────────────────────────────────┐
│  PHẦN 2: TẢI DATASET (Cell 6)               │
│  • Download Multi30K (EN-FR)                │
│  • Lấy train/val/test splits                │
└─────────────────────────────────────────────┘
           ⬇
┌─────────────────────────────────────────────┐
│  PHẦN 3-4: TIỀN XỬ LÝ (Cells 8, 10)        │
│  • Tokenization (spaCy)                     │
│  • Build Vocabulary với freq threshold      │
│  • Numericalization                         │
└─────────────────────────────────────────────┘
           ⬇
┌─────────────────────────────────────────────┐
│  PHẦN 5: DATASET & DATALOADER (Cell 12)    │
│  • TranslationDataset class                 │
│  • collate_fn (padding + sorting by length) │
│  • Batch size = 64                          │
└─────────────────────────────────────────────┘
           ⬇
┌─────────────────────────────────────────────┐
│  PHẦN 6: MODEL ARCHITECTURE (Cells 14-18)  │
│  • Encoder (LSTM + packing)                 │
│  • Decoder (LSTM)                           │
│  • Seq2Seq (attention-ready)                │
└─────────────────────────────────────────────┘
           ⬇
┌─────────────────────────────────────────────┐
│  PHẦN 7-8: TRAINING (Cells 20, 22) ⭐      │
│  • train_epoch() function                   │
│  • evaluate() function                      │
│  • Training loop với early stopping         │
└─────────────────────────────────────────────┘
           ⬇
┌─────────────────────────────────────────────┐
│  PHẦN 9-13: ĐÁNH GIÁ & INFERENCE (Cells 24-32) │
│  • Plot training curves                     │
│  • Inference function                       │
│  • Calculate BLEU score                     │
└─────────────────────────────────────────────┘
           ⬇
┌─────────────────────────────────────────────┐
│  PHẦN 14-17: ATTENTION MODEL (Cells 40-46) │
│  • Attention mechanism                      │
│  • DecoderWithAttention                     │
│  • Seq2SeqWithAttention                     │
│  • Training & Comparison                    │
└─────────────────────────────────────────────┘
```

---

## II. KIỂM TRA PHẦN TRAINING ✅

### 2.1 Hàm `train_epoch()` (Cell 20)

```python
def train_epoch(model, iterator, optimizer, criterion, clip, device, tf_ratio):
    model.train()                    # ✅ Set model to training mode
    epoch_loss = 0.0

    for src, src_lengths, trg in tqdm(iterator, ...):
        src, trg = src.to(device), trg.to(device)  # ✅ Move to device
        
        optimizer.zero_grad()        # ✅ Clear gradients
        output = model(src, src_lengths, trg, 
                      teacher_forcing_ratio=tf_ratio)  # [B, T, V]
        
        # ✅ Reshape để tính loss (loại bỏ <sos>)
        output = output[:, 1:, :].reshape(-1, V)
        trg    = trg[:, 1:].reshape(-1)
        
        loss = criterion(output, trg)  # ✅ Tính loss
        loss.backward()                # ✅ Backpropagation
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # ✅ Clip gradient
        optimizer.step()               # ✅ Update weights
        
        epoch_loss += loss.item()      # ✅ Accumulate loss
    
    return epoch_loss / len(iterator)  # ✅ Return average loss
```

**✅ Điểm tốt:**
- ✓ `model.train()` - đảm bảo dropout & batch norm hoạt động
- ✓ `optimizer.zero_grad()` - xóa gradient cũ trước backward
- ✓ `clip_grad_norm_()` - tránh exploding gradient
- ✓ Reshape output để loại bỏ `<sos>` token
- ✓ Accumulate loss chính xác

---

### 2.2 Hàm `evaluate()` (Cell 20)

```python
@torch.no_grad()                    # ✅ No gradient computation
def evaluate(model, iterator, criterion, device):
    model.eval()                    # ✅ Set model to eval mode
    epoch_loss = 0.0
    
    for src, src_lengths, trg in tqdm(iterator, ...):
        src, trg = src.to(device), trg.to(device)
        
        output = model(src, src_lengths, trg, teacher_forcing_ratio=0)
        # ✅ teacher_forcing_ratio=0 → use predictions, not ground truth
        
        output = output[:, 1:, :].reshape(-1, V)
        trg    = trg[:, 1:].reshape(-1)
        
        epoch_loss += criterion(output, trg).item()
    
    return epoch_loss / len(iterator)
```

**✅ Điểm tốt:**
- ✓ `@torch.no_grad()` - không tính gradient (tiết kiệm memory)
- ✓ `model.eval()` - tắt dropout, batch norm
- ✓ `teacher_forcing_ratio=0` - đánh giá thực tế mô hình dự đoán gì

---

### 2.3 Cấu hình Loss, Optimizer, Scheduler (Cell 20)

```python
PAD_IDX = fr_vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
# ✅ ignore_index tránh tính loss trên padding tokens

optimizer = optim.Adam(model.parameters(), lr=1e-3)
# ✅ Learning rate = 0.001 (hợp lý cho transformer-like model)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)
# ✅ Giảm learning rate nếu val loss không cải thiện

CLIP = 1.0
# ✅ Gradient clipping threshold
```

**✅ Phân tích:**
| Thành phần | Giá trị | Nhận xét |
|-----------|--------|---------|
| Loss | CrossEntropyLoss | ✅ Chuẩn cho NMT |
| LR | 0.001 | ✅ Phù hợp |
| Optimizer | Adam | ✅ Adaptive learning rate |
| Scheduler | ReduceLROnPlateau | ✅ Adaptive scheduling |
| Grad Clip | 1.0 | ✅ Tránh gradient explosion |

---

### 2.4 Training Loop (Cell 22)

```python
N_EPOCHS = 20
PATIENCE = 3  # Early stopping
best_valid_loss = float("inf")
patience_counter = 0

for epoch in range(N_EPOCHS):
    # 1️⃣ Training
    train_loss = train_epoch(model, train_loader, optimizer, criterion,
                            clip=CLIP, device=device, tf_ratio=TEACHER_FORCING_RATIO)
    # ✅ Training với teacher forcing (tỉ lệ 0.5)
    
    # 2️⃣ Validation
    valid_loss = evaluate(model, val_loader, criterion, device)
    # ✅ Validation không có teacher forcing (tỉ lệ 0)
    
    # 3️⃣ Learning rate scheduling
    scheduler.step(valid_loss)
    # ✅ Điều chỉnh learning rate dựa trên val loss
    
    # 4️⃣ Early stopping
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pth")  # ✅ Lưu best model
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            break  # ✅ Dừng nếu không cải thiện trong 3 epochs
```

**✅ Điểm tốt:**
- ✓ Early stopping với `patience=3`
- ✓ Lưu best model (`best_model.pth`)
- ✓ Learning rate scheduling
- ✓ Tính cả train loss và val loss

**⚠️ Có thể cải thiện:**
- Có thể thêm test loss tracking
- Có thể lưu checkpoint mỗi epoch cho debug

---

## III. PHÂN TÍCH KIẾN TRÚC MODEL

### 3.1 Encoder (Cell 14)

```python
class Encoder(nn.Module):
    def forward(self, src, src_lengths, return_outputs=False):
        embedded = self.dropout(self.embedding(src))  # [B, T, emb_dim]
        
        # ✅ Pack padded sequence (bỏ padding)
        packed = pack_padded_sequence(embedded, src_lengths, 
                                     batch_first=True, enforce_sorted=True)
        
        packed_output, (hidden, cell) = self.lstm(packed)
        
        # ✅ Unpack nếu cần cho attention
        if return_outputs:
            encoder_outputs, _ = pad_packed_sequence(packed_output, batch_first=True)
            return hidden, cell, encoder_outputs
        
        return hidden, cell, None
```

**✅ Điểm tốt:**
- ✓ Sử dụng `pack_padded_sequence` (hiệu quả hơn)
- ✓ Return `encoder_outputs` cho attention
- ✓ Dropout trên embedding

---

### 3.2 Decoder (Cell 16)

```python
class Decoder(nn.Module):
    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1)  # [B] → [B, 1]
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))  # [B, vocab_size]
        return prediction, hidden, cell
```

**✅ Điểm tốt:**
- ✓ Step-by-step decoding
- ✓ Maintains hidden state

---

### 3.3 Seq2Seq (Cell 18)

```python
class Seq2Seq(nn.Module):
    def forward(self, src, src_lengths, trg, teacher_forcing_ratio=0.5):
        hidden, cell, _ = self.encoder(src, src_lengths)
        
        inp = trg[:, 0]  # <sos> token
        for t in range(1, trg_len):
            out, hidden, cell = self.decoder(inp, hidden, cell)
            
            # ✅ Teacher forcing
            if random.random() < teacher_forcing_ratio:
                inp = trg[:, t]  # Use ground truth
            else:
                inp = out.argmax(1)  # Use prediction
```

**✅ Điểm tốt:**
- ✓ Chuẩn teacher forcing
- ✓ Support inference (ratio=0)

---

## IV. HYPERPARAMETERS SUMMARY

| Tham số | Giá trị | Mục đích |
|---------|--------|---------|
| **Input/Output Dim** | 10000 / 10000 | Vocab size |
| **Embedding Dim** | 256 | Token embedding dimension |
| **Hidden Dim** | 512 | LSTM hidden state size |
| **LSTM Layers** | 2 | Số stacked LSTM layers |
| **Dropout** | 0.4 | Regularization |
| **Batch Size** | 64 | Mini-batch size |
| **Learning Rate** | 0.001 | Adam learning rate |
| **Teacher Forcing** | 0.5 | Tỉ lệ dùng ground truth |
| **Gradient Clip** | 1.0 | Clip grad norm |
| **Scheduler** | ReduceLROnPlateau | Adaptive LR |
| **Patience (Early Stop)** | 3 | Epochs không cải thiện |
| **Max Epochs** | 20 | Max training epochs |

---

## V. RÚT RA ĐƯỢC GÌ TỪ NOTEBOOK?

### 5.1 Kiến thức NMT (Neural Machine Translation)

1. **Seq2Seq Architecture**: Encoder-Decoder với LSTM
2. **Teacher Forcing**: Accelerate training nhưng có exposure bias
3. **Attention Mechanism**: Giải quyết bottleneck của context vector
4. **BLEU Score**: Metric đánh giá translation quality
5. **Early Stopping**: Tránh overfitting

### 5.2 Kỹ thuật PyTorch

1. **Pack/Unpack**: Xử lý variable-length sequences hiệu quả
2. **Gradient Clipping**: Tránh exploding gradient
3. **Learning Rate Scheduling**: Adaptive learning rate
4. **Checkpoint Saving**: Lưu best model
5. **Teacher Forcing**: Dynamic training strategy

### 5.3 Best Practices

| Kỹ thuật | Lợi ích |
|---------|--------|
| `model.train()` / `model.eval()` | Điều khiển dropout/batchnorm |
| `@torch.no_grad()` | Tiết kiệm memory, tốc độ nhanh |
| `ignore_index` trong loss | Bỏ qua padding tokens |
| `clip_grad_norm_()` | Tránh gradient explosion |
| Early stopping | Tránh overfitting |
| Learning rate scheduling | Converge tốt hơn |

---

## VI. CÓ CHUẨN KHÔNG?

### ✅ **CÓ CHUẨN (Điểm tốt)**

1. **Loss function**: Đúng (CrossEntropyLoss + ignore_index)
2. **Optimizer**: Hợp lý (Adam + scheduler)
3. **Training loop**: Chuẩn (train/eval, early stopping)
4. **Gradient clipping**: Có (CLIP = 1.0)
5. **Checkpoint saving**: Có
6. **Teacher forcing**: Chuẩn (tỉ lệ 0.5)
7. **Data handling**: Tốt (padding, sorting by length)

### ⚠️ **CÓ THỂ CẢI THIỆN**

1. **Validation frequency**: Chỉ validate 1 lần/epoch
   - Có thể validate mỗi K batch để catch overfitting sớm hơn

2. **Metrics logging**: Chỉ log loss
   - Có thể log thêm BLEU, accuracy trên validation

3. **Hyperparameter tuning**: Chưa có ablation study
   - Có thể thử các tỉ lệ teacher forcing khác
   - Có thể thử learning rate khác

4. **Batch size handling**: Fixed batch size
   - Có thể dùng dynamic batching để tối ưu GPU memory

5. **Data shuffling**: Shuffle train set nhưng không validation/test
   - ✅ Đúng (nhưng comment thêm để rõ)

6. **Random seed**: Chưa set seed
   - Nên set `torch.manual_seed()` để reproducibility

---

## VII. KIẾN THỨC CORE TỪNG CELL

| Cell | Nội dung | Học được gì |
|------|---------|------------|
| 4 | Setup dependencies | PyTorch installation, device management |
| 6 | Download dataset | Data loading từ URL, gzip handling |
| 8 | Tokenization | spaCy tokenizer, preprocessing |
| 10 | Vocabulary building | Frequency-based vocab, numericalization |
| 12 | Dataset & DataLoader | Custom Dataset class, collate_fn |
| 14 | Encoder | pack_padded_sequence, LSTM |
| 16 | Decoder | Step-by-step decoding |
| 18 | Seq2Seq | Teacher forcing, inference |
| **20** | **Training setup** | **Loss, optimizer, scheduler** |
| **22** | **Training loop** | **Early stopping, checkpoint** |
| 24 | Visualization | Matplotlib, training curves |
| 28 | Inference | Greedy decoding |
| 30 | BLEU score | nltk bleu_score |
| 40 | Attention | Attention mechanism |
| 42 | DecoderWithAttention | Attention + decoder |
| 44 | Seq2SeqWithAttention | Full attention model |

---

## KẾT LUẬN

**✅ Code training CÓ CHUẨN!**

Notebook này làm tốt:
- ✓ Chuẩn architecture (Encoder-Decoder)
- ✓ Chuẩn training loop
- ✓ Chuẩn hyperparameters
- ✓ Có early stopping & checkpoint
- ✓ Có learning rate scheduling

**Có thể upgrade:**
- Thêm metrics tracking (BLEU on validation)
- Thêm reproducibility (set seed)
- Thêm ablation studies
- Thêm validation frequency

---

