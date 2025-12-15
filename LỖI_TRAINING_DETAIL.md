# 🐛 LỖI TRONG TRAINING VÀ CÁCH FIX

## **🔴 LỖI: ValueError - not enough values to unpack (expected 3, got 2)**

### **1. VỊ TRÍ LỖI**
Cell 14 - Encoder class - method `forward()`

### **2. NGUYÊN NHÂN**

**Code cũ (SAIIII):**
```python
def forward(self, src, src_lengths, return_outputs=False):
    # ...
    packed_output, (hidden, cell) = self.lstm(packed)
    
    _, (hidden, cell) = self.lstm(packed)  # ❌ Chạy LSTM LẦN 2!
    return hidden, cell  # ❌ Return chỉ 2 giá trị
```

**Vấn đề:**
- Chạy LSTM 2 lần → lãng phí tính toán
- Return chỉ 2 giá trị (hidden, cell)
- Nhưng training code cố unpack 3 giá trị:
  ```python
  hidden, cell, _ = self.encoder(src, src_lengths, return_outputs=False)
  ```
  → **ValueError: not enough values to unpack (expected 3, got 2)**

---

### **3. GIẢI PHÁP**

**Code mới (ĐÚNG):**
```python
def forward(self, src, src_lengths, return_outputs=False):
    # src: [B, src_len], src_lengths: [B] (sorted desc)
    embedded = self.dropout(self.embedding(src))  # [B, src_len, emb_dim]

    packed = pack_padded_sequence(
        embedded,
        lengths=src_lengths.cpu(),
        batch_first=True,
        enforce_sorted=True
    )

    packed_output, (hidden, cell) = self.lstm(packed)
    
    # ✅ Chỉ unpack nếu cần (cho attention model)
    if return_outputs:
        from torch.nn.utils.rnn import pad_packed_sequence
        encoder_outputs, _ = pad_packed_sequence(packed_output, batch_first=True)
        return hidden, cell, encoder_outputs  # ✅ Return 3 giá trị
    
    return hidden, cell, None  # ✅ Return 3 giá trị (giá trị cuối là None)
```

---

## **✅ THAY ĐỔI ĐÃ ĐƯỢC THỰC HIỆN**

✓ Xóa dòng LSTM chạy lần 2 (dòng `_, (hidden, cell) = self.lstm(packed)`)
✓ Thêm logic `if return_outputs` để unpack khi cần
✓ Return **luôn 3 giá trị** (hidden, cell, encoder_outputs hoặc None)
✓ Tương thích với training code: `hidden, cell, _ = self.encoder(...)`
✓ Tương thích với Attention model: sử dụng `encoder_outputs` khi cần

---

## **📝 DÒNG CÓ LỖI CHI TIẾT**

| Dòng | Code | Lỗi |
|------|------|-----|
| 391-392 | `packed_output, (hidden, cell) = self.lstm(packed)`  `_, (hidden, cell) = self.lstm(packed)` | ❌ Chạy LSTM 2 lần |
| 393 | `return hidden, cell` | ❌ Return 2 giá trị, nhưng code cầu 3 |

---

## **🎯 KẾT QUẢ SAU KHI FIX**

Encoder sẽ:
- ✅ Chạy LSTM **đúng 1 lần** (tiết kiệm 50% computation)
- ✅ Return **3 giá trị luôn** (hidden, cell, encoder_outputs/None)
- ✅ Tương thích với Seq2Seq baseline
- ✅ Tương thích với Seq2SeqWithAttention

---

## **TEST LẠI**

Bây giờ hãy chạy lại Cell 22 (Training) xem lỗi có biến mất không! 🚀

