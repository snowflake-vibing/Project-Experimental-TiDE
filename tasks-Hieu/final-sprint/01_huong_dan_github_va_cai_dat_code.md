# 💻 BƯỚC 1: CLONE REPO, CÀI CODE ĐỀ XUẤT & PUSH 2 BRANCH (`feat/m07` & `feat/m08`)

> [!IMPORTANT]
> Repo gốc [`AIVIETNAM-AIO-dthphuc/TiDE`](https://github.com/AIVIETNAM-AIO-dthphuc/TiDE) (nhánh `main`) **đã được clone sạch sẵn** tại **`D:\PTDLKD\TiDE`** (chưa sửa gì).
> Bạn sẽ tự tạo **2 nhánh độc lập từ `main`** ngay bên trong `D:\PTDLKD\TiDE` để cài riêng từng đề xuất:
> * **`feat/m07`:** Low-Rank Factorized Dense Decoder Projection
> * **`feat/m08`:** Temporal-Smoothed Dense Decoder

---

## 0. Quy tắc tách nhánh Git bắt buộc

Mỗi đề xuất phải nằm trên **1 branch riêng** tách trực tiếp từ `main` để không lẫn code khi chạy đối chứng trên Kaggle:

```bash
cd D:\PTDLKD\TiDE

# Kiểm tra chắc chắn đang ở nhánh main và sạch sẽ
git status
git checkout main
git pull origin main
```

> [!CAUTION]
> **Tuyệt đối không** sửa code và commit/push trực tiếp lên nhánh `main`! Luôn đứng ở `main` rồi `git checkout -b feat/m07` (xong việc quay lại `git checkout main` rồi mới `git checkout -b feat/m08`).

---

## 🔧 PHẦN A: CÀI ĐẶT ĐỀ XUẤT `M07` TRÊN NHÁNH `feat/m07`

### A.1. Tạo nhánh `feat/m07`

```bash
cd D:\PTDLKD\TiDE
git checkout main
git checkout -b feat/m07
```

### A.2. Cập nhật `README.md` cho `feat/m07` (Bắt buộc)

Theo quy định nhóm, mỗi branch cải tiến phải có `README.md` ghi rõ **Title phương pháp** và **Sơ lược ý tưởng cải tiến**. Bạn có thể chèn mục dưới đây lên đầu file `README.md` trong nhánh `feat/m07`:

```markdown
# M07 — Low-Rank Factorized Dense Decoder Projection for TiDE

## Sơ lược ý tưởng cải tiến
- **Vấn đề cần cải thiện:** Tầng chiếu cuối của khối `Dense Decoder` trong TiDE gốc dùng hai ma trận dày đặc (`lin_b` và `lin_res` tại `models.py#L40-L47`, `#L133-L137`) nối trực tiếp từ không gian ẩn `d_h` sang kích thước phẳng `H * p`. Khi dự báo tầm xa (`H = 720`) hoặc `p` lớn (VD: `ETTh2` chọn `p = 32`; `Electricity` chọn `d_h = 1024, p = 8`), số lượng tham số tầng cuối bùng nổ rất mạnh (~11.8M tham số), đúng như hạn chế mà tác giả thừa nhận ở Mục 6 Conclusion.
- **Ý tưởng cải tiến:** Thay thế ma trận chiếu dày đặc `W` kích thước `(H * p, d_h)` ở khối cuối `Dense Decoder` (cho cả nhánh chính `lin_b` và đường tắt `lin_res`) bằng tích của hai ma trận hạng thấp đi qua một nút thắt `r << min(d_h, H * p)`: `Linear(d_h -> r, bias=False)` nối tiếp `Linear(r -> H * p)`, với mốc mặc định `r = 64` (cho phép cấu hình qua cờ `--decoder_rank`).
- **Cơ sở khoa học & Kỳ vọng:** Kế thừa Sainath et al. (ICASSP 2013) về phân rã tầng Linear ngõ ra kích thước lớn khi train từ đầu và LoRA (Hu et al., ICLR 2022) về hạng nội tại thấp. Giảm ~85.4% tham số khối cuối ngay trên `ETTh1/ETTh2 H=96` (`r=64, d_h=512, p=32`), giảm ~92.6% trên `Electricity H=720`, giảm FLOPs, giữ nguyên tốc độ suy luận One-Shot `O(1)` và đóng vai trò điều chuẩn ngầm chống Overfitting.
```

### A.3. Chỉnh sửa `models.py` (tạo lớp Low-Rank và gắn vào `self.decoder`)

Quan sát code gốc trong `D:\PTDLKD\TiDE\models.py`:
* `MLPResidual` nằm tại **`models.py#L29-L61`** (`self.lin_a`, `self.lin_b`, `self.lin_res`, `self.lnorm`, `self.dropout`).
* `_make_dnn_residual` nằm tại **`models.py#L64-L81`**.
* Khởi tạo `self.decoder` của `TideModel.__init__` nằm tại **`models.py#L133-L140`**:
  ```python
  self.decoder = _make_dnn_residual(
      model_config.get('hidden_dims')[:-1]
      + [
          model_config.get('decoder_output_dim') * self.pred_len,
      ],
      layer_norm=layer_norm,
      dropout_rate=dropout_rate,
  )
  ```

**Cách sửa chi tiết trong `models.py`:**

1. Ngay bên dưới `class MLPResidual` (sau dòng 61), thêm hai lớp `LowRankDense` và `LowRankMLPResidual`:

```python
class LowRankDense(keras.layers.Layer):
  """Two-stage factorized linear projection: input_dim -> rank -> output_dim."""

  def __init__(self, rank, output_dim, use_bias=True):
    super(LowRankDense, self).__init__()
    self.proj_down = tf.keras.layers.Dense(
        rank,
        activation=None,
        use_bias=False,
    )
    self.proj_up = tf.keras.layers.Dense(
        output_dim,
        activation=None,
        use_bias=use_bias,
    )

  def call(self, inputs):
    return self.proj_up(self.proj_down(inputs))


class LowRankMLPResidual(keras.layers.Layer):
  """MLPResidual with low-rank factorization on output and skip projections."""

  def __init__(
      self, hidden_dim, output_dim, rank, layer_norm=False, dropout_rate=0.0
  ):
    super(LowRankMLPResidual, self).__init__()
    self.lin_a = tf.keras.layers.Dense(
        hidden_dim,
        activation='relu',
    )
    self.lin_b = LowRankDense(rank, output_dim, use_bias=True)
    self.lin_res = LowRankDense(rank, output_dim, use_bias=True)
    if layer_norm:
      self.lnorm = tf.keras.layers.LayerNormalization()
    self.layer_norm = layer_norm
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, inputs):
    """Call method."""
    h_state = self.lin_a(inputs)
    out = self.lin_b(h_state)
    out = self.dropout(out)
    res = self.lin_res(inputs)
    if self.layer_norm:
      return self.lnorm(out + res)
    return out + res
```

2. Thêm hàm helper `_make_low_rank_decoder` ngay dưới `_make_dnn_residual` (sau dòng 81) — xử lý đầy đủ **cả 2 trường hợp** `num_layers == 1` (`len(decoder_dims) < 2`) và `num_layers >= 2`:

```python
def _make_low_rank_decoder(
    decoder_dims, decoder_rank=None, layer_norm=False, dropout_rate=0.0
):
  """Build Dense Decoder with optional Low-Rank final projection block."""
  if decoder_rank is None or decoder_rank <= 0:
    return _make_dnn_residual(
        decoder_dims,
        layer_norm=layer_norm,
        dropout_rate=dropout_rate,
    )
  if len(decoder_dims) < 2:
    # Trường hợp num_layers == 1: chỉ có 1 lớp chiếu trực tiếp -> thay bằng LowRankDense
    return LowRankDense(decoder_rank, decoder_dims[-1], use_bias=True)

  layers = []
  # Các khối trung gian giữ nguyên MLPResidual chuẩn (như d_h -> d_h)
  for i, hdim in enumerate(decoder_dims[:-2]):
    layers.append(
        MLPResidual(
            hdim,
            decoder_dims[i + 1],
            layer_norm=layer_norm,
            dropout_rate=dropout_rate,
        )
    )
  # Chỉ khối cuối cùng (d_h -> H * p) dùng LowRankMLPResidual
  layers.append(
      LowRankMLPResidual(
          decoder_dims[-2],
          decoder_dims[-1],
          rank=decoder_rank,
          layer_norm=layer_norm,
          dropout_rate=dropout_rate,
      )
  )
  return keras.Sequential(layers)
```

3. Tại `TideModel.__init__` (**`models.py#L133-L140`**), thay khối khởi tạo `self.decoder` bằng:

```python
    decoder_dims = model_config.get('hidden_dims')[:-1] + [
        model_config.get('decoder_output_dim') * self.pred_len,
    ]
    self.decoder = _make_low_rank_decoder(
        decoder_dims,
        decoder_rank=model_config.get('decoder_rank', None),
        layer_norm=layer_norm,
        dropout_rate=dropout_rate,
    )
```

> [!TIP]
> **Kiểm tra nhanh phép tính tham số trên `ETTh1 / ETTh2` ở `H = 96` (`hidden_size = 512, p = 32` $\Rightarrow H \cdot p = 3{,}072$):**
> * **TiDE gốc (`lin_b` + `lin_res`):** `2 * (512 * 3,072) + 2 * 3,072 = 3,151,872` tham số (riêng trọng số ma trận là `3,145,728`).
> * **M07 với `r = 64`:** `2 * (512 * 64 + 64 * 3,072) + 2 * 3,072 = 464,896` tham số (trọng số ma trận còn `458,752`, **giảm ~85.4%** ngay ở `H=96`; ở `Electricity H=720` giảm ~92.6%).

### A.4. Chỉnh sửa `train.py` (thêm cờ `--decoder_rank`)

1. Tại vùng khai báo flags (**`train.py#L65-L78`**), thêm cờ mới:
   ```python
   flags.DEFINE_integer(
       'decoder_rank',
       64,
       'Low-rank bottleneck dimension for final Dense Decoder projection (<=0 to disable).',
   )
   ```
2. Tại `_run_config()` (**`train.py#L165-L172`**), thêm `'decoder_rank'` vào tuple `keys` để cơ chế `--resume` kiểm tra khớp cấu hình `run_config.json`:
   ```python
   keys = (
       'dataset', 'random_seed', 'hist_len', 'pred_len', 'batch_size',
       'learning_rate', 'num_layers', 'hidden_size', 'decoder_output_dim',
       'final_decoder_hidden', 'decoder_rank', 'datetime_col', 'ts_cols',
       'num_cov_cols', 'cat_cov_cols', 'epoch_len', 'normalize', 'holiday',
       'permute', 'transform', 'layer_norm', 'dropout_rate', 'num_split',
       'min_num_epochs', 'patience', 'train_epochs',
   )
   ```
3. Tại `model_config` (**`train.py#L340-L347`**), truyền thêm `'decoder_rank'`:
   ```python
   model_config = {
       'model_type': 'dnn',
       'hidden_dims': [FLAGS.hidden_size] * FLAGS.num_layers,
       'time_encoder_dims': [64, 4],
       'decoder_output_dim': FLAGS.decoder_output_dim,
       'final_decoder_hidden': FLAGS.final_decoder_hidden,
       'decoder_rank': FLAGS.decoder_rank,
       'batch_size': dtl.batch_size,
   }
   ```

### A.5. Kiểm thử nhanh Local & Push `feat/m07`

Chạy đoạn test nhanh trong `D:\PTDLKD\TiDE` để xác nhận không lỗi shape trước khi push:

```bash
python -c "
import numpy as np, tensorflow as tf, models
for rank in [0, 64]:
    cfg = {'hidden_dims': [512, 512], 'time_encoder_dims': [64, 4],
           'decoder_output_dim': 32, 'final_decoder_hidden': 16, 'decoder_rank': rank}
    m = models.TideModel(cfg, pred_len=96, cat_sizes=[], num_ts=7, transform=True, layer_norm=True, dropout_rate=0.2)
    B, L, H, F = 8, 720, 96, 8
    past = [tf.zeros((B, L)), tf.zeros((F, L)), tf.zeros((0, L), dtype=tf.int32)]
    fut  = [tf.zeros((F, H)), tf.zeros((0, H), dtype=tf.int32)]
    out = m((past, fut, tf.zeros((B,), dtype=tf.int32)), training=False)
    dec_params = sum(int(np.prod(v.shape)) for v in m.decoder.trainable_variables)
    print(f'rank={rank} -> out={out.shape}, decoder_params={dec_params:,}')
"
```

Khi thấy `out=(8, 96)` và `decoder_params` giảm đúng mức (~85%), push nhánh `feat/m07` lên GitHub:

```bash
cd D:\PTDLKD\TiDE
git status
git add README.md models.py train.py
git commit -m "feat(m07): add Low-Rank Factorized Dense Decoder Projection (--decoder_rank)"
git push -u origin feat/m07
```

---

## 🔧 PHẦN B: CÀI ĐẶT ĐỀ XUẤT `M08` TRÊN NHÁNH `feat/m08`

### B.1. Quay về `main` và tạo nhánh `feat/m08`

> [!WARNING]
> Nhớ quay về `main` trước khi tách nhánh `feat/m08` để code M08 hoàn toàn sạch, không dính code của M07!

```bash
cd D:\PTDLKD\TiDE
git checkout main
git checkout -b feat/m08
```

### B.2. Cập nhật `README.md` cho `feat/m08` (Bắt buộc)

```markdown
# M08 — Temporal-Smoothed Dense Decoder for TiDE

## Sơ lược ý tưởng cải tiến
- **Vấn đề cần cải thiện:** Do cơ chế giải mã One-Shot tách ma trận chiếu tầng cuối `Dense Decoder` thành các lát cắt độc lập cho từng bước thời gian `t = 1..H`, ngay sau thao tác `tf.reshape` (`models.py#L208-L210`) thành ma trận `D` kích thước `(B, p, H)`, giữa các cột bối cảnh liền kề `d_t` và `d_{t+1}` không có tương tác cục bộ trực tiếp, có thể gây dao động răng cưa (jitter) trên chuỗi nhạy pha.
- **Ý tưởng cải tiến:** Chèn một lớp tích chập 1D theo chiều sâu siêu nhẹ (`DepthwiseConv1D` với `kernel_size = 3`, `padding = 'same'`, `depth_multiplier = 1`) kết hợp kết nối dư (`residual`) ngay sau phép `Reshape` của `Dense Decoder` trước khi nối ghép ngoại sinh tương lai để đưa sang `Temporal Decoder` (`models.py#L211`).
- **Cơ sở khoa học & Kỳ vọng:** Kế thừa TCN (Bai et al., 2018) về hiệu quả lọc cục bộ trên chuỗi thời gian. Chi phí tham số gần như bằng 0 (chỉ thêm `(k + 1) * p = 4 * 32 = 128` tham số ở `ETTh1/ETTh2`), giữ nguyên tốc độ song song `O(1)` trên GPU và giúp làm mượt chuỗi dự báo, kỳ vọng cải thiện MSE/MAE.
```

### B.3. Chỉnh sửa `models.py` (chèn `DepthwiseConv1D` sau `tf.reshape`)

1. Trong `TideModel.__init__` (ngay sau dòng **`models.py#L140`**), khởi tạo sẵn lớp làm mịn cục bộ để TensorFlow Checkpoint (`train.py#L370-L385`) theo dõi đầy đủ biến trainable:

```python
    self.smooth_kernel_size = model_config.get('smooth_kernel_size', 3)
    if self.smooth_kernel_size and self.smooth_kernel_size > 0:
      self.temporal_smoother = tf.keras.layers.DepthwiseConv1D(
          kernel_size=self.smooth_kernel_size,
          padding='same',
          depth_multiplier=1,
          use_bias=True,
          depthwise_initializer='zeros',
          bias_initializer='zeros',
      )
    else:
      self.temporal_smoother = None
```

> [!NOTE]
> Khởi tạo trọng số bằng `zeros` kèm phép cộng dư `d_trans + self.temporal_smoother(d_trans)` giúp tại Epoch 0 đầu ra của mô hình giống hệt TiDE gốc (ổn định bước khởi đầu), sau đó mô hình tự học mức làm mịn tối ưu qua `kernel_size = 3`.

2. Trong `TideModel.call` ngay sau dòng **`models.py#L208-L210`** (`decoder_out` đang có shape `(B, p, H)`):

> [!CAUTION]
> **Bẫy Shape cực kỳ quan trọng trong TensorFlow:**
> `tf.keras.layers.DepthwiseConv1D` mặc định dùng chuẩn `channels_last`, tức là đòi input shape **`(Batch, Length, Channels) = (B, H, p)`**. Trong khi `decoder_out` ở dòng 209 có shape **`(B, p, H)`** (`channels_first`)! Vì vậy **bắt buộc** phải `tf.transpose` sang `(0, 2, 1)` trước khi qua `DepthwiseConv1D`, rồi transpose ngược lại `(0, 2, 1)` để trả về `(B, p, H)` khớp với dòng 211!

```python
    decoder_out = tf.reshape(
        decoder_out, [bsize, -1, self.pred_len]
    )  # batch x d x H
    if self.temporal_smoother is not None:
      d_trans = tf.transpose(decoder_out, (0, 2, 1))  # (B, H, p)
      d_trans = d_trans + self.temporal_smoother(d_trans)
      decoder_out = tf.transpose(d_trans, (0, 2, 1))  # (B, p, H)
    final_in = tf.concat([decoder_out, enc_fut], axis=1)
```

### B.4. Chỉnh sửa `train.py` (thêm cờ `--smooth_kernel_size`)

1. Tại vùng khai báo flags (**`train.py#L65-L78`**):
   ```python
   flags.DEFINE_integer(
       'smooth_kernel_size',
       3,
       'DepthwiseConv1D kernel size for temporal smoothing after Unflatten (<=0 to disable).',
   )
   ```
2. Tại `_run_config()` (**`train.py#L165-L172`**), thêm `'smooth_kernel_size'` vào `keys`.
3. Tại `model_config` (**`train.py#L340-L347`**), thêm `'smooth_kernel_size': FLAGS.smooth_kernel_size`.

### B.5. Kiểm thử nhanh Local & Push `feat/m08`

```bash
python -c "
import numpy as np, tensorflow as tf, models
cfg = {'hidden_dims': [512, 512], 'time_encoder_dims': [64, 4],
       'decoder_output_dim': 32, 'final_decoder_hidden': 16, 'smooth_kernel_size': 3}
m = models.TideModel(cfg, pred_len=96, cat_sizes=[], num_ts=7, transform=True, layer_norm=True, dropout_rate=0.2)
B, L, H, F = 8, 720, 96, 8
past = [tf.zeros((B, L)), tf.zeros((F, L)), tf.zeros((0, L), dtype=tf.int32)]
fut  = [tf.zeros((F, H)), tf.zeros((0, H), dtype=tf.int32)]
out = m((past, fut, tf.zeros((B,), dtype=tf.int32)), training=False)
sm_params = sum(int(np.prod(v.shape)) for v in m.temporal_smoother.trainable_variables)
print(f'out={out.shape}, temporal_smoother_params={sm_params}')
"
```

Khi xác nhận `out=(8, 96)` và `temporal_smoother_params=128`, push nhánh `feat/m08`:

```bash
cd D:\PTDLKD\TiDE
git status
git add README.md models.py train.py
git commit -m "feat(m08): add Temporal-Smoothed DepthwiseConv1D after Unflatten (--smooth_kernel_size)"
git push -u origin feat/m08
```

---

## 📩 BƯỚC CUỐI CỦA BƯỚC 1: NHẮN NHÓM TRƯỞNG DUYỆT

Ngay sau khi push xong `feat/m07` và `feat/m08` lên GitHub `AIVIETNAM-AIO-dthphuc/TiDE`, **nhắn tin báo ngay cho nhóm trưởng** để kiểm tra và duyệt 2 nhánh trước khi bạn chạy notebook trên Kaggle!
