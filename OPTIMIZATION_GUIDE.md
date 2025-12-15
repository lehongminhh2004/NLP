# ⚡ PHÂN TÍCH VÀ TỐI ƯU PERFORMANCE - TRAINING CHẬM

## I. CÁC VẤN ĐỀ CHÍNH GẠY CHẬM

### 🔴 **1. COLLATE_FN - Sorting Each Batch (MAJOR)**

**Vấn đề:**
```python
def collate_fn(batch, pad_idx=0):
    src_list, trg_list = zip(*batch)
    src_lengths = torch.tensor([len(s) for s in src_list], dtype=torch.long)
    
    # ⚠️ BOTTLENECK: Sorting mỗi batch O(B log B)
    src_lengths, sort_idx = src_lengths.sort(descending=True)
    src_batch = src_batch[sort_idx]
    trg_batch = trg_batch[sort_idx]
    
    return src_batch, src_lengths, trg_batch
```

**Tác động:**
- Mỗi batch (64 samples) phải sort → **O(64 log 64) = ~384 operations**
- Train: 29,000 samples → ~453 batches/epoch
- **Total: ~175,392 sort operations mỗi epoch** 🐢

**Giải pháp:**
1. **Bucket Sampler**: Nhóm câu theo độ dài TRƯỚC khi vào DataLoader
2. **Không sort**: Dữ liệu đã pre-sorted
3. Hoặc dùng `BucketBatchSampler`

---

### 🔴 **2. DataLoader - Thiếu Optimization (MAJOR)**

**Vấn đề:**
```python
train_loader = DataLoader(train_dataset, 
                         batch_size=BATCH_SIZE, 
                         shuffle=True,  
                         collate_fn=collate_fn)
# ❌ Thiếu: num_workers, pin_memory, prefetch_factor
```

**Thiếu optimizations:**
| Tham số | Hiện tại | Khuyến nghị | Lợi ích |
|---------|----------|------------|---------|
| `num_workers` | 0 (main thread) | 4-8 | Parallel data loading |
| `pin_memory` | False | True | Faster GPU transfer |
| `prefetch_factor` | 2 | 4-8 | Pre-load batches |
| `persistent_workers` | N/A | True | Tái sử dụng processes |

**Tác động:**
- Mỗi batch phải load từ CPU → GPU sequentially
- Không parallel data processing
- GPU chờ data trong khi CPU load → **GPU idle time lớn**

**Tính toán:**
- Batch load time: ~0.5s
- 453 batches/epoch × 0.5s = **~226 seconds** cho chỉ data loading! 😱

---

### 🟠 **3. Model Architecture - Quá Phức Tạp (MEDIUM)**

**Hiện tại:**
```python
EMBEDDING_DIM = 256       # 256 dimensions
HIDDEN_DIM = 512          # 512 hidden units
N_LAYERS = 2              # 2 LSTM layers
```

**Số parameters:**
```
Encoder:
  - Embedding: 10,000 vocab × 256 = 2.56M
  - LSTM (2 layers): 256 + 512 × 4 × 512 × 2 = 4.19M
  Total: ~6.75M

Decoder:
  - Embedding: 10,000 × 256 = 2.56M
  - LSTM: 4.19M
  Total: ~6.75M

Total Model: ~13.5M parameters
```

**Tác động:**
- Mỗi forward pass: 13.5M × 4 bytes = 54MB memory
- Backward pass: 54MB × 2 = 108MB gradient
- **Tổng: ~162MB per batch** (chưa tính optimizer state)

**Giải pháp:**
- Giảm HIDDEN_DIM từ 512 → 256 (giảm 75% computation)
- Hoặc giảm EMBEDDING_DIM từ 256 → 128

---

### 🟠 **4. Pack/Unpack Sequence - Có Overhead (MEDIUM)**

**Vấn đề:**
```python
packed = pack_padded_sequence(embedded, lengths=src_lengths, 
                              batch_first=True, enforce_sorted=True)
packed_output, (hidden, cell) = self.lstm(packed)
encoder_outputs, _ = pad_packed_sequence(packed_output, batch_first=True)
```

**Tác động:**
- Pack: Create mask + sparse data structure
- LSTM: Chạy trên sparse data (não efficient cho CUDA)
- Unpack: Restore padding
- **Overhead: ~10-15% tính toán**

**Khi nào hữu dụng:**
- ✅ Khi variance độ dài lớn (ví dụ: 5 → 100 tokens)
- ❌ Khi variance nhỏ (ví dụ: 50 → 65 tokens)

---

### 🟡 **5. Teacher Forcing - Random Check Mỗi Step (MINOR)**

**Vấn đề:**
```python
for t in range(1, trg_len):
    out, hidden, cell = self.decoder(inp, hidden, cell)
    outputs[:, t, :] = out
    
    # ⚠️ Random number generation mỗi timestep
    teacher_force = random.random() < teacher_forcing_ratio
    top1 = out.argmax(1)
    inp = trg[:, t] if teacher_force else top1
```

**Tác động:**
- Mỗi target token (trung bình ~20): `random.random()` call
- 453 batches × 64 samples × 20 tokens × random() = **579,840 random calls/epoch**
- ~2-3% overhead

**Giải pháp:**
- Pre-generate random masks trước
- Vectorize: `teacher_force = torch.rand(B) < ratio`

---

### 🟡 **6. Không Sử Dụng Mixed Precision (MINOR)**

**Vấn đề:**
```python
# ❌ Tất cả computation sử dụng float32
output = model(src, src_lengths, trg, ...)
loss = criterion(output, trg)
```

**Có thể cải thiện:**
- Float32 computation chậm hơn float16
- GPU (nhất là RTX series) tối ưu cho float16

**Tác động:**
- ~1.5-2x speedup nếu dùng `torch.autocast()`

---

## II. QUICK FIXES (Áp dụng ngay)

### ✅ **Fix 1: Tối ưu DataLoader**

```python
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,              # ✅ Parallel loading
    pin_memory=True,            # ✅ GPU memory pinning
    prefetch_factor=4,          # ✅ Pre-load batches
    persistent_workers=True,    # ✅ Reuse workers
    drop_last=True              # ✅ Avoid small last batch
)
```

**Dự kiến speedup: ~40-50%** ⚡

---

### ✅ **Fix 2: Giảm Model Size**

```python
# Before
EMBEDDING_DIM = 256
HIDDEN_DIM = 512

# After
EMBEDDING_DIM = 128  # ✅ Giảm 50%
HIDDEN_DIM = 256     # ✅ Giảm 50%
```

**Dự kiến speedup: ~60-70%** ⚡⚡

---

### ✅ **Fix 3: Vectorize Teacher Forcing**

```python
# Before
for t in range(1, trg_len):
    out, hidden, cell = self.decoder(inp, hidden, cell)
    teacher_force = random.random() < teacher_forcing_ratio  # ❌ Slow
    inp = trg[:, t] if teacher_force else out.argmax(1)

# After
def forward_with_teacher_forcing(self, src, src_lengths, trg, tf_ratio=0.5):
    # Generate mask một lần
    max_len = trg.size(1)
    tf_mask = torch.rand(trg.size(0), max_len) < tf_ratio  # [B, T]
    
    for t in range(1, max_len):
        out, hidden, cell = self.decoder(inp, hidden, cell)
        # Vectorized
        inp = torch.where(tf_mask[:, t], trg[:, t], out.argmax(1))
```

**Dự kiến speedup: ~2-3%** (nhỏ nhưng cleaner)

---

### ✅ **Fix 4: Enable Mixed Precision**

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

def train_epoch(model, iterator, optimizer, criterion, clip, device, tf_ratio):
    model.train()
    epoch_loss = 0.0
    
    for src, src_lengths, trg in iterator:
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        
        # ✅ Use mixed precision
        with autocast():
            output = model(src, src_lengths, trg, teacher_forcing_ratio=tf_ratio)
            loss = criterion(output.reshape(-1, output.size(-1)), 
                           trg[:, 1:].reshape(-1))
        
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)
```

**Dự kiến speedup: ~1.3-1.8x** ⚡⚡⚡

---

## III. ADVANCED FIXES (Cải thiện sâu)

### 🚀 **Advanced 1: Bucket Sampler (Loại Sorting)**

```python
class BucketBatchSampler:
    """Nhóm sequences theo length bucket → không cần sort/batch"""
    def __init__(self, data_lengths, batch_size, num_buckets=10):
        # Pre-sort data theo length
        # Nhóm vào buckets
        # Mỗi batch từ cùng bucket (variance nhỏ)
    
    def __iter__(self):
        for bucket in self.buckets:
            for i in range(0, len(bucket), self.batch_size):
                yield bucket[i:i+self.batch_size]

# Usage
sampler = BucketBatchSampler(src_lengths, batch_size=64, num_buckets=10)
train_loader = DataLoader(train_dataset, batch_sampler=sampler, 
                         num_workers=4, pin_memory=True)
```

**Dự kiến speedup: ~20-30%** (loại sorting overhead)

---

### 🚀 **Advanced 2: Compile Model (PyTorch 2.0+)**

```python
if torch.__version__ >= "2.0":
    model = torch.compile(model, mode="reduce-overhead")
    # ✅ JIT compile model
```

**Dự kiến speedup: ~1.2-1.5x**

---

### 🚀 **Advanced 3: Gradient Accumulation (nếu OOM)**

```python
accumulation_steps = 4

for epoch in range(N_EPOCHS):
    for batch_idx, (src, src_lengths, trg) in enumerate(train_loader):
        output = model(src, src_lengths, trg, tf_ratio=TEACHER_FORCING_RATIO)
        loss = criterion(output.reshape(-1, output.size(-1)), 
                        trg[:, 1:].reshape(-1))
        
        loss.backward()
        
        # Update mỗi 4 batches (nhưng data loading parallel)
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
            optimizer.zero_grad()
```

**Benefit:** Fit larger batches mà không OOM

---

## IV. QUICK DIAGNOSTIC

Để xác định bottleneck chính, thêm code này:

```python
import time

def train_epoch_with_profiling(model, iterator, optimizer, criterion, 
                               clip, device, tf_ratio):
    model.train()
    epoch_loss = 0.0
    
    data_load_time = 0
    forward_time = 0
    backward_time = 0
    
    start_data = time.time()
    
    for src, src_lengths, trg in tqdm(iterator):
        data_load_time += time.time() - start_data
        src, trg = src.to(device), trg.to(device)
        
        # Forward
        start_forward = time.time()
        output = model(src, src_lengths, trg, teacher_forcing_ratio=tf_ratio)
        V = output.size(-1)
        
        output = output[:, 1:, :].reshape(-1, V)
        trg = trg[:, 1:].reshape(-1)
        loss = criterion(output, trg)
        forward_time += time.time() - start_forward
        
        # Backward
        start_backward = time.time()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        optimizer.zero_grad()
        backward_time += time.time() - start_backward
        
        epoch_loss += loss.item()
        start_data = time.time()
    
    total_time = data_load_time + forward_time + backward_time
    print(f"\n⏱️  Profiling Results:")
    print(f"  • Data Loading:  {data_load_time:.2f}s ({data_load_time/total_time*100:.1f}%)")
    print(f"  • Forward Pass:  {forward_time:.2f}s ({forward_time/total_time*100:.1f}%)")
    print(f"  • Backward Pass: {backward_time:.2f}s ({backward_time/total_time*100:.1f}%)")
    print(f"  • Total Time:    {total_time:.2f}s")
    
    return epoch_loss / len(iterator)
```

---

## V. RECOMMENDED OPTIMIZATION STRATEGY

**Thứ tự áp dụng (từ dễ → khó):**

| Thứ tự | Fix | Speedup | Độ khó | Thời gian |
|--------|-----|---------|--------|-----------|
| 1️⃣ | DataLoader (num_workers) | 40-50% | ⭐ | 2 phút |
| 2️⃣ | Giảm model size | 60-70% | ⭐ | 5 phút |
| 3️⃣ | Mixed Precision | 1.3-1.8x | ⭐⭐ | 10 phút |
| 4️⃣ | Bucket Sampler | 20-30% | ⭐⭐ | 30 phút |
| 5️⃣ | torch.compile | 1.2-1.5x | ⭐⭐ | 5 phút |

**Tổng hợp: ~2-3x tổng speedup với vài bước đơn giản!** 🚀

---

## VI. EXAMPLE: OPTIMIZED CODE

```python
from torch.cuda.amp import autocast, GradScaler

# 1. Tối ưu DataLoader
train_loader = DataLoader(
    train_dataset, 
    batch_size=64,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=4,
    persistent_workers=True,
)

# 2. Giảm model size
EMBEDDING_DIM = 128   # từ 256
HIDDEN_DIM = 256      # từ 512

encoder = Encoder(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT)
decoder = Decoder(OUTPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT)
model = Seq2Seq(encoder, decoder, device).to(device)

# 3. Mixed precision
scaler = GradScaler()

# 4. Training loop
def train_epoch_optimized(model, iterator, optimizer, criterion, clip, device, tf_ratio):
    model.train()
    epoch_loss = 0.0
    
    for src, src_lengths, trg in tqdm(iterator):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        
        # Mixed precision
        with autocast():
            output = model(src, src_lengths, trg, teacher_forcing_ratio=tf_ratio)
            V = output.size(-1)
            loss = criterion(output[:, 1:, :].reshape(-1, V), 
                           trg[:, 1:].reshape(-1))
        
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)
```

**Dự kiến:**
- Trước: 30 phút/epoch
- Sau: ~10-15 phút/epoch ⚡⚡⚡

---

